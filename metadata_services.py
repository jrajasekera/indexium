"""Metadata planning and backup services for smart metadata management."""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import ffmpeg


def _normalize_person_list(names: Iterable[str]) -> list[str]:
    """Return a cleaned, sorted, and deduplicated list of person names."""
    cleaned = {name.strip() for name in names if name and name.strip()}
    return sorted(cleaned, key=lambda name: name.lower())


def extract_people_from_comment(comment: str | None) -> list[str]:
    """Extract person tags from an ffmpeg comment string."""
    if not comment:
        return []

    people_segment = None
    marker = "People:"
    if marker in comment:
        after_marker = comment.split(marker, 1)[1]
        people_segment = after_marker
        for terminator in ("\n", ";", "|"):
            if terminator in people_segment:
                people_segment = people_segment.split(terminator, 1)[0]
    if people_segment is None:
        return []

    names = [part.strip() for part in people_segment.split(",")]
    return _normalize_person_list(names)


@dataclass(slots=True)
class PlanStatistics:
    total_files: int
    safe_count: int
    warning_count: int
    danger_count: int
    blocked_count: int
    total_tags_to_add: int
    will_overwrite_custom: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanItem:
    file_hash: str
    file_path: str | None
    file_name: str | None
    file_extension: str | None
    db_people: list[str]
    existing_people: list[str]
    result_people: list[str]
    tags_to_add: list[str]
    tags_to_remove: list[str]
    existing_comment: str | None
    result_comment: str
    risk_level: str
    can_update: bool
    issues: list[str] = field(default_factory=list)
    probe_error: str | None = None
    metadata_only_people: list[str] = field(default_factory=list)
    will_overwrite_comment: bool = False
    overwrites_custom_comment: bool = False
    issue_codes: list[str] = field(default_factory=list)
    tag_count: int = 0
    new_tag_count: int = 0
    file_modified_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requires_update"] = self.requires_update
        return data

    @property
    def requires_update(self) -> bool:
        if not self.can_update:
            return True
        existing = (self.existing_comment or "").strip()
        return existing != self.result_comment


@dataclass(slots=True)
class MetadataPlan:
    items: list[PlanItem]
    statistics: PlanStatistics
    categories: dict[str, list[PlanItem]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "statistics": self.statistics.to_dict(),
            "categories": {
                category: [item.file_hash for item in items]
                for category, items in self.categories.items()
            },
        }


@dataclass(slots=True)
class BackupRecord:
    id: int
    operation_item_id: int
    file_hash: str
    file_path: str
    backup_timestamp: datetime
    original_comment: str | None
    original_metadata_json: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["backup_timestamp"] = self.backup_timestamp.isoformat()
        return data


class MetadataPlanner:
    """Service responsible for building metadata change plans."""

    def __init__(
        self,
        ffmpeg_module: Any = ffmpeg,
        database_path: str | None = None,
        max_workers: int | None = None,
    ):
        self.ffmpeg = ffmpeg_module
        self._ffmpeg_error = getattr(ffmpeg_module, "Error", ffmpeg.Error)
        self._database_path = database_path
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = min(8, max(1, cpu_count))
        self._max_workers = max(1, max_workers)
        self._logger = logging.getLogger(__name__)
        self._cache_table_ready = False
        self._comment_cache_lock = threading.Lock()

    def generate_plan(
        self,
        conn: sqlite3.Connection,
        file_hashes: Sequence[str] | None = None,
        include_blocked: bool = True,
    ) -> MetadataPlan:
        query = [
            "SELECT DISTINCT file_hash FROM faces WHERE person_name IS NOT NULL",
        ]
        params: list[Any] = []
        if file_hashes:
            placeholders = ",".join("?" for _ in file_hashes)
            query.append(f"AND file_hash IN ({placeholders})")
            params.extend(file_hashes)

        query.append("ORDER BY file_hash")
        rows = conn.execute("\n".join(query), params).fetchall()
        target_hashes: list[str] = [
            row[0] if isinstance(row, tuple) else row["file_hash"] for row in rows
        ]

        items: list[PlanItem] = []

        should_parallelize = self._database_path and len(target_hashes) > 1 and self._max_workers > 1

        if should_parallelize:
            worker_count = min(self._max_workers, len(target_hashes))

            def _build_with_new_connection(file_hash: str) -> PlanItem | None:
                local_conn: sqlite3.Connection | None = None
                try:
                    assert self._database_path is not None
                    local_conn = sqlite3.connect(self._database_path)
                    local_conn.row_factory = sqlite3.Row
                    return self._build_plan_item(local_conn, file_hash)
                except Exception:  # pragma: no cover - defensive logging
                    self._logger.exception("Failed building metadata plan item for %s", file_hash)
                    return None
                finally:
                    if local_conn is not None:
                        local_conn.close()

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for item in executor.map(_build_with_new_connection, target_hashes, chunksize=1):
                    if item is None:
                        continue
                    if include_blocked or item.can_update:
                        items.append(item)
        else:
            for file_hash in target_hashes:
                item = self._build_plan_item(conn, file_hash)
                if include_blocked or item.can_update:
                    items.append(item)

        categories = self.categorize_items(items)
        statistics = self.calculate_statistics(items)
        return MetadataPlan(items=items, statistics=statistics, categories=categories)

    def _ensure_comment_cache_table(self, conn: sqlite3.Connection) -> None:
        """Create the metadata comment cache table if missing."""
        if self._cache_table_ready:
            return

        with self._comment_cache_lock:
            if self._cache_table_ready:
                return
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_comment_cache (
                    file_hash TEXT PRIMARY KEY,
                    comment TEXT,
                    file_mtime REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
            self._cache_table_ready = True

    def categorize_items(self, items: Sequence[PlanItem]) -> dict[str, list[PlanItem]]:
        buckets: dict[str, list[PlanItem]] = {
            "safe": [],
            "warning": [],
            "danger": [],
            "blocked": [],
        }
        for item in items:
            if not item.can_update:
                buckets["blocked"].append(item)
            buckets.setdefault(item.risk_level, []).append(item)
        # Remove empty categories for cleaner JSON
        return {key: value for key, value in buckets.items() if value}

    def calculate_statistics(self, items: Sequence[PlanItem]) -> PlanStatistics:
        total_files = len(items)
        safe_count = sum(1 for item in items if item.risk_level == "safe")
        warning_count = sum(1 for item in items if item.risk_level == "warning")
        danger_count = sum(1 for item in items if item.risk_level == "danger")
        blocked_count = sum(1 for item in items if not item.can_update)
        total_tags_to_add = sum(len(item.tags_to_add) for item in items)
        will_overwrite_custom = sum(1 for item in items if item.overwrites_custom_comment)
        return PlanStatistics(
            total_files=total_files,
            safe_count=safe_count,
            warning_count=warning_count,
            danger_count=danger_count,
            blocked_count=blocked_count,
            total_tags_to_add=total_tags_to_add,
            will_overwrite_custom=will_overwrite_custom,
        )

    def filter_items(
        self,
        items: Sequence[PlanItem],
        filters: dict[str, Any] | None = None,
    ) -> list[PlanItem]:
        if not filters:
            return list(items)

        filtered = list(items)
        risk_levels = filters.get("risk_levels")
        if risk_levels:
            normalized = {level.lower() for level in risk_levels}

            def risk_match(plan_item: PlanItem) -> bool:
                if "blocked" in normalized and not plan_item.can_update:
                    return True
                return plan_item.risk_level in normalized

            filtered = [item for item in filtered if risk_match(item)]

        requires_update = filters.get("requires_update")
        if requires_update is not None:
            filtered = [item for item in filtered if item.requires_update == bool(requires_update)]

        can_update = filters.get("can_update")
        if can_update is not None:
            filtered = [item for item in filtered if item.can_update == bool(can_update)]

        tag_range = filters.get("tag_count") or {}
        min_tags = tag_range.get("min")
        max_tags = tag_range.get("max")
        if min_tags is not None:
            filtered = [item for item in filtered if item.tag_count >= int(min_tags)]
        if max_tags is not None:
            filtered = [item for item in filtered if item.tag_count <= int(max_tags)]

        issue_codes = filters.get("issue_codes")
        if issue_codes:
            wanted = {code.lower() for code in issue_codes}
            filtered = [
                item
                for item in filtered
                if wanted.intersection(code.lower() for code in item.issue_codes)
            ]

        file_types = filters.get("file_types")
        if file_types:
            normalized = {ext.lower().lstrip(".") for ext in file_types}
            filtered = [
                item
                for item in filtered
                if item.file_extension and item.file_extension.lower().lstrip(".") in normalized
            ]

        search = (filters.get("search") or "").strip().lower()
        if search:
            filtered = [
                item
                for item in filtered
                if search in (item.file_name or "").lower()
                or any(search in person.lower() for person in item.result_people)
                or search in (item.file_hash or "").lower()
            ]

        return filtered

    def sort_items(
        self,
        items: Sequence[PlanItem],
        sort_by: str | None = None,
        direction: str = "asc",
    ) -> list[PlanItem]:
        if not sort_by:
            return list(items)

        reverse = direction.lower() == "desc"

        def risk_priority(level: str) -> int:
            return {
                "danger": 3,
                "warning": 2,
                "safe": 1,
                "blocked": 0,
            }.get(level, 0)

        if sort_by == "risk":
            return sorted(
                items,
                key=lambda item: (risk_priority(item.risk_level), item.file_name or item.file_hash),
                reverse=reverse,
            )
        if sort_by == "tag_count":
            return sorted(
                items,
                key=lambda item: (item.tag_count, item.file_name or item.file_hash),
                reverse=reverse,
            )
        if sort_by == "modified":
            return sorted(
                items,
                key=lambda item: (item.file_modified_time or 0.0, item.file_name or item.file_hash),
                reverse=reverse,
            )
        # Default alphabetical sort
        return sorted(items, key=lambda item: (item.file_name or item.file_hash), reverse=reverse)

    def update_item_with_custom_people(
        self,
        item: PlanItem,
        custom_people: Sequence[str],
    ) -> PlanItem:
        result_people = _normalize_person_list(custom_people)
        if not result_people:
            result_people = list(item.result_people)

        tags_to_add = sorted(set(result_people) - set(item.existing_people), key=str.lower)
        tags_to_remove = sorted(set(item.existing_people) - set(result_people), key=str.lower)
        metadata_only_people = sorted(
            set(item.existing_people) - set(item.db_people), key=str.lower
        )

        issues = list(item.issues)
        issue_codes = list(item.issue_codes)
        risk_level = item.risk_level

        # Update risk/issue notes based on manual edits
        if tags_to_remove:
            removal_issue = "Removing people present in existing metadata"
            if removal_issue not in issues:
                issues.append(removal_issue)
            risk_level = "warning"
            if "tag_removal" not in issue_codes:
                issue_codes.append("tag_removal")

        result_comment = f"People: {', '.join(result_people)}"
        will_overwrite_comment = (item.existing_comment or "").strip() != result_comment
        overwrites_custom_comment = item.overwrites_custom_comment

        return PlanItem(
            file_hash=item.file_hash,
            file_path=item.file_path,
            file_name=item.file_name,
            file_extension=item.file_extension,
            db_people=item.db_people,
            existing_people=item.existing_people,
            result_people=result_people,
            tags_to_add=tags_to_add,
            tags_to_remove=tags_to_remove,
            existing_comment=item.existing_comment,
            result_comment=result_comment,
            risk_level=risk_level,
            can_update=item.can_update,
            issues=issues,
            probe_error=item.probe_error,
            metadata_only_people=metadata_only_people,
            will_overwrite_comment=will_overwrite_comment,
            overwrites_custom_comment=overwrites_custom_comment,
            issue_codes=issue_codes,
            tag_count=len(result_people),
            new_tag_count=len(tags_to_add),
            file_modified_time=item.file_modified_time,
        )

    def _build_plan_item(self, conn: sqlite3.Connection, file_hash: str) -> PlanItem:
        db_people = _normalize_person_list(
            name_row[0]
            for name_row in conn.execute(
                "SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL",
                (file_hash,),
            )
        )

        video_row = conn.execute(
            "SELECT last_known_filepath FROM scanned_files WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        file_path = video_row[0] if video_row else None
        file_name = os.path.basename(file_path) if file_path else None
        file_extension = Path(file_path).suffix if file_path else None
        can_update = bool(file_path and Path(file_path).exists())

        existing_comment: str | None = None
        existing_people: list[str] = []
        probe_error: str | None = None
        file_modified_time: float | None = None

        cache_valid = False
        if can_update:
            assert file_path is not None  # can_update implies file_path exists
            self._ensure_comment_cache_table(conn)
            try:
                file_modified_time = os.path.getmtime(file_path)
            except OSError:
                file_modified_time = None

            cache_row = conn.execute(
                "SELECT comment, file_mtime FROM metadata_comment_cache WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()

            if cache_row is not None:
                cached_comment = (
                    cache_row[0] if isinstance(cache_row, tuple) else cache_row["comment"]
                )
                cached_mtime = (
                    cache_row[1] if isinstance(cache_row, tuple) else cache_row["file_mtime"]
                )
                if (
                    file_modified_time is not None
                    and cached_mtime is not None
                    and abs(float(cached_mtime) - float(file_modified_time)) < 0.5
                ):
                    existing_comment = cached_comment
                    existing_people = extract_people_from_comment(existing_comment)
                    cache_valid = True
                elif file_modified_time is None and cached_mtime is None:
                    existing_comment = cached_comment
                    existing_people = extract_people_from_comment(existing_comment)
                    cache_valid = True

            if not cache_valid:
                try:
                    probe = self.ffmpeg.probe(file_path)
                    existing_comment = probe.get("format", {}).get("tags", {}).get("comment")
                    existing_people = extract_people_from_comment(existing_comment)
                except (
                    self._ffmpeg_error
                ) as exc:  # pragma: no cover - exercised in integration scenarios
                    probe_error = (
                        exc.stderr.decode("utf8") if getattr(exc, "stderr", None) else str(exc)
                    )
                    existing_comment = ""
                    existing_people = []
                except Exception as exc:  # pragma: no cover - defensive fallback
                    probe_error = str(exc)
                    existing_comment = ""
                    existing_people = []

                if file_modified_time is None:
                    try:
                        file_modified_time = os.path.getmtime(file_path)
                    except OSError:
                        file_modified_time = None

                try:
                    conn.execute(
                        """
                        INSERT INTO metadata_comment_cache (file_hash, comment, file_mtime)
                        VALUES (?, ?, ?)
                        ON CONFLICT(file_hash) DO UPDATE SET
                            comment = excluded.comment,
                            file_mtime = excluded.file_mtime,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (file_hash, existing_comment, file_modified_time),
                    )
                    conn.commit()
                except Exception:  # pragma: no cover - cache writes are best-effort
                    self._logger.exception(
                        "Failed updating metadata comment cache for %s", file_hash
                    )

        result_people = _normalize_person_list(list(set(db_people).union(existing_people)))
        metadata_only_people = sorted(set(existing_people) - set(db_people), key=str.lower)
        tags_to_add = sorted(set(db_people) - set(existing_people), key=str.lower)
        tags_to_remove = sorted(set(existing_people) - set(result_people), key=str.lower)
        result_comment = f"People: {', '.join(result_people)}" if result_people else ""

        existing_comment_value = (existing_comment or "").strip()
        will_overwrite_comment = (
            can_update and bool(existing_comment_value) and existing_comment_value != result_comment
        )
        overwrites_custom_comment = bool(
            can_update
            and existing_comment_value
            and not existing_comment_value.startswith("People:")
            and existing_comment_value != result_comment
        )

        risk_level = "safe"
        issues: list[str] = []
        issue_codes: list[str] = []
        if not can_update:
            risk_level = "danger"
            issues.append("Video file path is unavailable")
            issue_codes.append("missing_file")
        elif overwrites_custom_comment:
            risk_level = "danger"
            issues.append("Will overwrite existing custom comment")
            issue_codes.append("custom_comment")
        elif will_overwrite_comment:
            risk_level = "warning"
            issues.append("Will replace existing metadata comment")
            issue_codes.append("overwrite_comment")

        if probe_error:
            issues.append("ffprobe error: " + probe_error.splitlines()[0])
            issue_codes.append("ffprobe_error")

        if metadata_only_people:
            issue_codes.append("metadata_only_people")

        return PlanItem(
            file_hash=file_hash,
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            db_people=db_people,
            existing_people=existing_people,
            result_people=result_people,
            tags_to_add=tags_to_add,
            tags_to_remove=tags_to_remove,
            existing_comment=existing_comment,
            result_comment=result_comment,
            risk_level=risk_level,
            can_update=can_update,
            issues=issues,
            probe_error=probe_error,
            metadata_only_people=metadata_only_people,
            will_overwrite_comment=will_overwrite_comment,
            overwrites_custom_comment=overwrites_custom_comment,
            issue_codes=issue_codes,
            tag_count=len(result_people),
            new_tag_count=len(tags_to_add),
            file_modified_time=file_modified_time,
        )


class BackupManager:
    """Service for capturing and restoring video metadata backups."""

    def __init__(self, ffmpeg_module: Any = ffmpeg):
        self.ffmpeg = ffmpeg_module

    def create_backup(
        self,
        conn: sqlite3.Connection,
        file_hash: str,
        file_path: str,
        operation_item_id: int,
    ) -> BackupRecord:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        probe = self.ffmpeg.probe(file_path)
        tags = probe.get("format", {}).get("tags", {}) or {}
        original_comment = tags.get("comment")
        original_metadata_json = json.dumps(tags)

        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO metadata_history (operation_item_id, file_hash, original_comment, original_metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (operation_item_id, file_hash, original_comment, original_metadata_json),
        )
        conn.commit()
        backup_id = cursor.lastrowid
        row = cursor.execute(
            """
            SELECT id, operation_item_id, file_hash, backup_timestamp,
                   original_comment, original_metadata_json
            FROM metadata_history WHERE id = ?
            """,
            (backup_id,),
        ).fetchone()
        timestamp = datetime.fromisoformat(row[3])
        return BackupRecord(
            id=row[0],
            operation_item_id=row[1],
            file_hash=row[2],
            file_path=file_path,
            backup_timestamp=timestamp,
            original_comment=row[4],
            original_metadata_json=row[5],
        )

    def restore_backup(self, conn: sqlite3.Connection, operation_item_id: int) -> bool:
        row = conn.execute(
            """
            SELECT h.file_hash, h.original_comment, h.original_metadata_json, i.file_path
            FROM metadata_history h
            JOIN metadata_operation_items i ON h.operation_item_id = i.id
            WHERE h.operation_item_id = ?
            """,
            (operation_item_id,),
        ).fetchone()
        if not row:
            return False

        _file_hash, original_comment, _metadata_json, file_path = row
        if not file_path or not Path(file_path).exists():
            return False

        temp_path = Path(file_path).with_name(f".restore_{Path(file_path).name}")
        comment_value = "" if original_comment is None else original_comment
        metadata_arg = f"comment={comment_value}"
        stream = self.ffmpeg.input(file_path)
        stream = self.ffmpeg.output(stream, str(temp_path), c="copy", metadata=metadata_arg)
        self.ffmpeg.run(stream, overwrite_output=True, quiet=True)
        os.replace(temp_path, file_path)
        return True

    def cleanup_old_backups(self, conn: sqlite3.Connection, days_to_keep: int = 90) -> int:
        threshold = datetime.utcnow() - timedelta(days=days_to_keep)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM metadata_history WHERE backup_timestamp < ?",
            (threshold.isoformat(sep=" ", timespec="seconds"),),
        )
        deleted = cursor.rowcount
        conn.commit()
        return deleted


@dataclass(slots=True)
class WriteOptions:
    """Options that control metadata writing behaviour."""

    dry_run: bool = False
    create_backups: bool = True
    overwrite_existing: bool = True


@dataclass
class _OperationRuntime:
    """In-memory control block for an active metadata write operation."""

    operation_id: int
    items: list[PlanItem]
    item_ids: list[int]
    pause_event: threading.Event
    cancel_event: threading.Event
    thread: threading.Thread | None = None


class MetadataWriter:
    """Service responsible for executing metadata writes asynchronously."""

    def __init__(
        self,
        database_path: str,
        ffmpeg_module: Any = ffmpeg,
        backup_manager: BackupManager | None = None,
    ) -> None:
        self.database_path = database_path
        self.ffmpeg = ffmpeg_module
        self.backup_manager = backup_manager or BackupManager(ffmpeg_module=ffmpeg_module)
        self._lock = threading.Lock()
        self._operations: dict[int, _OperationRuntime] = {}

    def start_operation(
        self,
        items: Sequence[PlanItem],
        options: WriteOptions | None = None,
        background: bool = True,
    ) -> int:
        """Create a metadata operation and dispatch background processing."""

        runnable_items = [item for item in items if item.can_update]
        if not runnable_items:
            raise ValueError("No writable items supplied to MetadataWriter")

        options = options or WriteOptions()

        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metadata_operations (operation_type, status, file_count)
                VALUES (?, ?, ?)
                """,
                ("write", "pending", len(items)),
            )
            operation_id = cursor.lastrowid
            assert operation_id is not None, "INSERT must return a lastrowid"

            item_ids: list[int] = []
            for plan_item in items:
                cursor.execute(
                    """
                    INSERT INTO metadata_operation_items (
                        operation_id,
                        file_hash,
                        file_path,
                        status,
                        previous_comment,
                        new_comment,
                        tags_added,
                        tags_removed
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        plan_item.file_hash,
                        plan_item.file_path or "",
                        "pending" if plan_item.can_update else "skipped",
                        plan_item.existing_comment,
                        plan_item.result_comment,
                        json.dumps(plan_item.tags_to_add),
                        json.dumps(plan_item.tags_to_remove),
                    ),
                )
                item_id = cursor.lastrowid
                assert item_id is not None, "INSERT must return a lastrowid"
                item_ids.append(item_id)

            conn.commit()

        runnable_ids = [
            item_id for item, item_id in zip(items, item_ids, strict=False) if item.can_update
        ]
        runtime = _OperationRuntime(
            operation_id=operation_id,
            items=list(runnable_items),
            item_ids=runnable_ids,
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
        )
        runtime.pause_event.set()

        with self._lock:
            self._operations[operation_id] = runtime

        if background:
            thread = threading.Thread(
                target=self._run_operation,
                args=(runtime, options),
                daemon=True,
                name=f"metadata-writer-{operation_id}",
            )
            runtime.thread = thread
            thread.start()
        else:
            self._run_operation(runtime, options)

        return operation_id

    def pause_operation(self, operation_id: int) -> bool:
        with self._lock:
            runtime = self._operations.get(operation_id)
        if not runtime:
            return False
        runtime.pause_event.clear()
        self._update_operation_status(operation_id, "paused")
        return True

    def resume_operation(self, operation_id: int) -> bool:
        with self._lock:
            runtime = self._operations.get(operation_id)
        if not runtime:
            return False
        runtime.pause_event.set()
        self._update_operation_status(operation_id, "in_progress")
        return True

    def cancel_operation(self, operation_id: int) -> bool:
        with self._lock:
            runtime = self._operations.get(operation_id)
        if not runtime:
            return False
        runtime.cancel_event.set()
        runtime.pause_event.set()
        self._update_operation_status(operation_id, "cancelling")
        return True

    def get_operation_status(self, operation_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            op_row = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if not op_row:
                return None

            item_rows = conn.execute(
                """
                SELECT id, file_hash, file_path, status, previous_comment, new_comment,
                       tags_added, tags_removed, error_message, processed_at
                FROM metadata_operation_items
                WHERE operation_id = ?
                ORDER BY id
                """,
                (operation_id,),
            ).fetchall()

        totals = {
            "success": 0,
            "failed": 0,
            "pending": 0,
            "skipped": 0,
        }
        items: list[dict[str, Any]] = []
        for row in item_rows:
            status = row["status"]
            if status == "success":
                totals["success"] += 1
            elif status == "failed":
                totals["failed"] += 1
            elif status == "skipped":
                totals["skipped"] += 1
            else:
                totals["pending"] += 1

            items.append(
                {
                    "id": row["id"],
                    "file_hash": row["file_hash"],
                    "file_path": row["file_path"],
                    "file_name": os.path.basename(row["file_path"] or row["file_hash"]),
                    "status": status,
                    "error_message": row["error_message"],
                    "processed_at": row["processed_at"],
                    "tags_added": json.loads(row["tags_added"] or "[]"),
                    "tags_removed": json.loads(row["tags_removed"] or "[]"),
                }
            )

        file_count = op_row["file_count"] or max(1, len(items))
        completed = totals["success"] + totals["failed"] + totals["skipped"]
        progress = completed / file_count if file_count else 1.0

        return {
            "operation_id": operation_id,
            "status": op_row["status"],
            "file_count": file_count,
            "success_count": op_row["success_count"],
            "failure_count": op_row["failure_count"],
            "pending_count": totals["pending"],
            "skipped_count": totals["skipped"],
            "error_message": op_row["error_message"],
            "started_at": op_row["started_at"],
            "completed_at": op_row["completed_at"],
            "progress": progress,
            "items": items,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_operation(self, runtime: _OperationRuntime, options: WriteOptions) -> None:
        operation_id = runtime.operation_id
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "UPDATE metadata_operations SET status = 'in_progress', started_at = COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id = ?",
                (operation_id,),
            )
            conn.commit()

        try:
            processed_ids: set[int] = set()
            for plan_item, item_id in zip(runtime.items, runtime.item_ids, strict=False):
                runtime.pause_event.wait()
                if runtime.cancel_event.is_set():
                    break

                try:
                    self._mark_item_status(item_id, "in_progress")
                    if options.dry_run:
                        time.sleep(0)
                    else:
                        self._write_single_item(plan_item, item_id, options)
                    self._record_success(operation_id, item_id)
                    processed_ids.add(item_id)
                except Exception as exc:  # noqa: BLE001
                    self._record_failure(operation_id, item_id, str(exc))
                    processed_ids.add(item_id)

                if runtime.cancel_event.is_set():
                    break

            if runtime.cancel_event.is_set():
                for pending_id in runtime.item_ids:
                    if pending_id not in processed_ids:
                        with sqlite3.connect(self.database_path) as conn:
                            conn.execute(
                                "UPDATE metadata_operation_items SET status = 'skipped', processed_at = CURRENT_TIMESTAMP WHERE id = ?",
                                (pending_id,),
                            )
                            conn.commit()
                self._update_operation_status(operation_id, "cancelled")
            else:
                self._update_operation_status(operation_id, "completed")
        finally:
            with self._lock:
                self._operations.pop(operation_id, None)

    def _update_operation_status(self, operation_id: int, status: str) -> None:
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "UPDATE metadata_operations SET status = ?, completed_at = CASE WHEN ? IN ('completed', 'cancelled') THEN CURRENT_TIMESTAMP ELSE completed_at END WHERE id = ?",
                (status, status, operation_id),
            )
            conn.commit()

    def _mark_item_status(self, item_id: int, status: str) -> None:
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "UPDATE metadata_operation_items SET status = ? WHERE id = ?",
                (status, item_id),
            )
            conn.commit()

    def _record_success(self, operation_id: int, item_id: int) -> None:
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                UPDATE metadata_operation_items
                SET status = 'success', processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (item_id,),
            )
            conn.execute(
                "UPDATE metadata_operations SET success_count = success_count + 1 WHERE id = ?",
                (operation_id,),
            )
            conn.commit()

    def _record_failure(self, operation_id: int, item_id: int, error_message: str) -> None:
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                UPDATE metadata_operation_items
                SET status = 'failed', processed_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE id = ?
                """,
                (error_message, item_id),
            )
            conn.execute(
                "UPDATE metadata_operations SET failure_count = failure_count + 1 WHERE id = ?",
                (operation_id,),
            )
            conn.commit()

    def _write_single_item(self, item: PlanItem, item_id: int, options: WriteOptions) -> None:
        if not item.file_path or not Path(item.file_path).exists():
            raise FileNotFoundError(f"Video file not found: {item.file_path}")

        temp_name = f".meta_{uuid.uuid4().hex}_{Path(item.file_path).name}"
        temp_path = str(Path(item.file_path).with_name(temp_name))

        if options.create_backups:
            with sqlite3.connect(self.database_path) as conn:
                self.backup_manager.create_backup(conn, item.file_hash, item.file_path, item_id)

        stream = self.ffmpeg.input(item.file_path)
        metadata_value = (
            item.result_comment
            if options.overwrite_existing
            else item.existing_comment or item.result_comment
        )
        metadata_arg = (
            f"comment={metadata_value}" if isinstance(metadata_value, str) else metadata_value
        )
        stream = self.ffmpeg.output(stream, temp_path, c="copy", metadata=metadata_arg)

        try:
            self.ffmpeg.run(stream, overwrite_output=True, quiet=True)
            os.replace(temp_path, item.file_path)
        except Exception:  # noqa: BLE001
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        if os.path.exists(temp_path):
            os.remove(temp_path)


class HistoryService:
    """Service for querying metadata operation history and performing rollbacks."""

    def __init__(
        self,
        database_path: str,
        backup_manager: BackupManager,
    ) -> None:
        self.database_path = database_path
        self.backup_manager = backup_manager

    def get_operations(
        self,
        filters: dict[str, Any] | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        filters = filters or {}
        page = max(1, page)
        per_page = max(1, per_page)

        conditions = []
        params: list[Any] = []

        statuses = filters.get("status")
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            conditions.append(f"status IN ({placeholders})")
            params.extend(statuses)

        start_date = filters.get("start_date")
        if start_date:
            conditions.append("date(started_at) >= date(?)")
            params.append(start_date)

        end_date = filters.get("end_date")
        if end_date:
            conditions.append("date(started_at) <= date(?)")
            params.append(end_date)

        search = filters.get("search")
        search_join = ""
        if search:
            search_join = "LEFT JOIN metadata_operation_items moi ON moi.operation_id = mo.id"
            like = f"%{search.lower()}%"
            conditions.append(
                "(lower(moi.file_hash) LIKE ? OR lower(moi.file_path) LIKE ? OR lower(mo.error_message) LIKE ?)"
            )
            params.extend([like, like, like])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute(
                f"SELECT COUNT(DISTINCT mo.id) FROM metadata_operations mo {search_join} {where_clause}",
                params,
            ).fetchone()[0]

            offset = (page - 1) * per_page
            rows = conn.execute(
                f"""
                SELECT DISTINCT mo.*
                FROM metadata_operations mo
                {search_join}
                {where_clause}
                ORDER BY mo.started_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [per_page, offset],
            ).fetchall()

        operations = [self._row_to_operation_summary(row) for row in rows]
        total_pages = max(1, math.ceil(total / per_page)) if per_page else 1

        return {
            "operations": operations,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": total,
                "total_pages": total_pages,
            },
        }

    def get_operation_details(self, operation_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            op_row = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if not op_row:
                return None

            item_rows = conn.execute(
                """
                SELECT moi.*, mh.backup_timestamp
                FROM metadata_operation_items moi
                LEFT JOIN metadata_history mh ON mh.operation_item_id = moi.id
                WHERE moi.operation_id = ?
                ORDER BY moi.id
                """,
                (operation_id,),
            ).fetchall()

        return {
            "operation": self._row_to_operation_summary(op_row),
            "items": [self._row_to_item_detail(row) for row in item_rows],
        }

    def rollback_operation(self, operation_id: int) -> dict[str, Any]:
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            op_row = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if not op_row:
                raise ValueError("Operation not found")

            if op_row["status"] not in {"completed", "failed", "cancelled", "rolled_back"}:
                raise ValueError("Operation is still running; pause or cancel before rolling back")

            item_rows = conn.execute(
                "SELECT id FROM metadata_operation_items WHERE operation_id = ?",
                (operation_id,),
            ).fetchall()

        restored = 0
        failed = 0

        with sqlite3.connect(self.database_path) as conn:
            for row in item_rows:
                item_id = row["id"]
                try:
                    success = self.backup_manager.restore_backup(conn, item_id)
                except Exception as exc:  # noqa: BLE001
                    success = False
                    conn.execute(
                        "UPDATE metadata_operation_items SET error_message = ? WHERE id = ?",
                        (f"Rollback error: {exc}", item_id),
                    )

                if success:
                    restored += 1
                    conn.execute(
                        "UPDATE metadata_operation_items SET status = 'rolled_back', processed_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (item_id,),
                    )
                else:
                    failed += 1
                    conn.execute(
                        "UPDATE metadata_operation_items SET status = 'rollback_failed', processed_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (item_id,),
                    )

            conn.execute(
                "UPDATE metadata_operations SET status = 'rolled_back', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (operation_id,),
            )
            conn.commit()

        return {
            "operation_id": operation_id,
            "restored": restored,
            "failed": failed,
        }

    def _row_to_operation_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "operation_type": row["operation_type"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "file_count": row["file_count"],
            "success_count": row["success_count"],
            "failure_count": row["failure_count"],
            "error_message": row["error_message"],
            "user_note": row["user_note"],
        }

    def _row_to_item_detail(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "file_hash": row["file_hash"],
            "file_path": row["file_path"],
            "status": row["status"],
            "previous_comment": row["previous_comment"],
            "new_comment": row["new_comment"],
            "tags_added": json.loads(row["tags_added"] or "[]"),
            "tags_removed": json.loads(row["tags_removed"] or "[]"),
            "error_message": row["error_message"],
            "processed_at": row["processed_at"],
            "backup_timestamp": row["backup_timestamp"],
        }
