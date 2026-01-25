"""NFO metadata services for Jellyfin-compatible metadata management."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

logger = logging.getLogger(__name__)


class NfoParseError(Exception):
    """Raised when NFO file cannot be parsed."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse {path}: {reason}")


@dataclass
class NfoActor:
    """Actor data extracted from NFO file.

    Note: raw_element is NOT cached. When writing, we always read the NFO
    fresh to get current raw elements, ensuring unknown children are preserved.
    """

    name: str
    source: str | None = None
    role: str | None = None
    type: str | None = None
    thumb: str | None = None
    raw_element: Any = field(default=None, repr=False)  # lxml Element, not cached


class NfoService:
    """Core service for reading and writing NFO files."""

    def find_nfo_path(self, video_path: str) -> str | None:
        """Find NFO file for a video, checking multiple naming conventions.

        Checks in order (returns first match):
        1. <video_name>.nfo / .NFO (video-specific, preferred)
        2. movie.nfo / Movie.nfo (shared, may apply to multiple videos)
        """
        base = Path(video_path).stem
        parent = Path(video_path).parent

        candidates = [
            parent / f"{base}.nfo",
            parent / f"{base}.NFO",
            parent / "movie.nfo",
            parent / "Movie.nfo",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def get_default_nfo_path(self, video_path: str) -> str:
        """Get the default NFO path for a video file (creates <video_name>.nfo)."""
        base = Path(video_path).stem
        parent = Path(video_path).parent
        return str(parent / f"{base}.nfo")

    def read_actors(self, nfo_path: str) -> list[NfoActor]:
        """Parse NFO XML, return all actors with full structure preserved.

        Raises:
            NfoParseError: If XML is malformed or unreadable
        """
        root, _ = self._read_xml(nfo_path)
        actors = []

        for actor_elem in root.findall("actor"):
            name_elem = actor_elem.find("name")
            if name_elem is None or not name_elem.text:
                continue

            actors.append(
                NfoActor(
                    name=name_elem.text.strip(),
                    source=actor_elem.get("source"),
                    role=self._get_child_text(actor_elem, "role"),
                    type=self._get_child_text(actor_elem, "type"),
                    thumb=self._get_child_text(actor_elem, "thumb"),
                    raw_element=actor_elem,
                )
            )

        return actors

    def _read_xml(self, nfo_path: str) -> tuple[etree._Element, str | None]:
        """Read NFO file, return (root element, detected encoding)."""
        try:
            with open(nfo_path, "rb") as f:
                raw = f.read()
        except OSError as e:
            raise NfoParseError(nfo_path, f"Cannot read file: {e}") from e

        # Detect BOM and encoding
        encoding = None
        if raw.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
            raw = raw[3:]
            encoding = "utf-8-sig"

        try:
            parser = etree.XMLParser(remove_blank_text=False, recover=True)
            root = etree.fromstring(raw, parser)

            if root is None:
                raise NfoParseError(nfo_path, "XML recovery failed")

            return root, encoding

        except etree.XMLSyntaxError as e:
            raise NfoParseError(nfo_path, f"XML syntax error: {e}") from e

    def write_actors(
        self,
        nfo_path: str,
        indexium_actors: list[str],
    ) -> None:
        """Update NFO: keep non-indexium actors, replace indexium actors.

        Raises:
            NfoParseError: If XML is malformed
        """
        root, encoding = self._read_xml(nfo_path)

        # Remove existing indexium actors
        for actor_elem in root.findall("actor"):
            if actor_elem.get("source") == "indexium":
                root.remove(actor_elem)

        # Add new indexium actors
        for name in sorted(set(indexium_actors), key=str.lower):
            actor_elem = etree.SubElement(root, "actor", source="indexium")
            name_elem = etree.SubElement(actor_elem, "name")
            name_elem.text = name

        self._write_xml(nfo_path, root, encoding)

    def _write_xml(self, nfo_path: str, root: etree._Element, encoding: str | None) -> None:
        """Write NFO file, preserving original encoding."""
        # Handle utf-8-sig specially (lxml doesn't understand this encoding name)
        lxml_encoding = "utf-8" if encoding == "utf-8-sig" else (encoding or "utf-8")

        xml_bytes = etree.tostring(
            root,
            encoding=lxml_encoding,
            xml_declaration=True,
            pretty_print=False,
        )

        with open(nfo_path, "wb") as f:
            if encoding == "utf-8-sig":
                f.write(b"\xef\xbb\xbf")  # Write BOM
            f.write(xml_bytes)

    def get_indexium_actors(self, nfo_path: str) -> list[str]:
        """Convenience: return only names where source='indexium'."""
        actors = self.read_actors(nfo_path)
        return [a.name for a in actors if a.source == "indexium"]

    @staticmethod
    def _get_child_text(elem: etree._Element, tag: str) -> str | None:
        """Get text content of child element, or None."""
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None


class NfoBackupManager:
    """Operation-scoped backup/restore for NFO files."""

    def find_backup_path(self, nfo_path: str, operation_id: int) -> str:
        """Return expected backup path for given NFO and operation."""
        return f"{nfo_path}.bak.{operation_id}"

    def create_backup(self, nfo_path: str, operation_id: int) -> str:
        """Copy NFO to operation-scoped backup, return backup path."""
        backup_path = self.find_backup_path(nfo_path, operation_id)
        shutil.copy2(nfo_path, backup_path)
        return backup_path

    def restore_backup(self, nfo_path: str, operation_id: int) -> bool:
        """Restore from operation-specific backup if exists. Returns success."""
        backup_path = self.find_backup_path(nfo_path, operation_id)
        if not Path(backup_path).exists():
            return False
        shutil.copy2(backup_path, nfo_path)
        return True

    def cleanup_backup(self, nfo_path: str, operation_id: int) -> None:
        """Remove operation-specific backup file."""
        backup_path = self.find_backup_path(nfo_path, operation_id)
        try:
            Path(backup_path).unlink()
        except FileNotFoundError:
            pass

    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Remove backup files older than max_age_days. Returns count removed."""
        # This would scan for .bak.* files and check mtime
        # Implementation deferred - needs directory scanning
        return 0


@dataclass
class NfoPlanItem:
    """Plan item with full UI/API compatibility.

    Maintains all fields expected by metadata_preview.html and app.py serialization.
    """

    # Core identifiers
    file_hash: str
    file_path: str | None
    file_name: str | None
    file_extension: str | None

    # NFO-specific (new)
    nfo_path: str | None  # None = no NFO file, skip

    # People data
    db_people: list[str]  # People tagged in Indexium DB
    existing_people: list[str]  # All existing Indexium actors (alias for UI)
    result_people: list[str]  # Final people list after operation
    tags_to_add: list[str]  # New people to add
    tags_to_remove: list[str]  # Indexium actors to remove

    # NFO-specific internal data
    existing_indexium_actors: list[str]  # Current <actor source="indexium">
    other_actors: list[NfoActor]  # Actors from Jellyfin/TMDb (preserved)

    # Comment fields (for UI compatibility - serialize actors)
    existing_comment: str | None  # Serialized existing Indexium actors
    result_comment: str  # Serialized result actors

    # Status and risk
    risk_level: str  # "safe", "warning", "danger", "blocked"
    can_update: bool  # False if no NFO file or parse error

    # Issues tracking
    issues: list[str] = field(default_factory=list)
    issue_codes: list[str] = field(default_factory=list)
    probe_error: str | None = None  # Now used for XML parse errors

    # UI display fields
    metadata_only_people: list[str] = field(default_factory=list)  # Non-indexium actors in NFO
    will_overwrite_comment: bool = False  # True if changing existing actors
    overwrites_custom_comment: bool = False  # True if danger level
    tag_count: int = 0  # len(result_people)
    new_tag_count: int = 0  # len(tags_to_add)
    file_modified_time: float | None = None  # NFO file mtime

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON API response."""
        data = asdict(self)
        data["requires_update"] = self.requires_update
        # Remove internal fields not needed by UI
        data.pop("other_actors", None)
        data.pop("existing_indexium_actors", None)
        return data

    @property
    def requires_update(self) -> bool:
        """True if this item needs to be written."""
        if not self.can_update:
            return False
        return bool(self.tags_to_add or self.tags_to_remove)


class NfoPlanner:
    """Builds change plans for NFO metadata updates."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nfo_service = NfoService()

    def build_plan(self, file_hashes: list[str]) -> list[NfoPlanItem]:
        """Build plan for given file hashes.

        Creates ONE plan item PER VIDEO (not per NFO) to preserve UI/API contract.
        """
        if not file_hashes:
            return []

        items = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for file_hash in file_hashes:
                item = self._build_item_for_hash(file_hash, conn)
                if item:
                    items.append(item)

        return items

    def _build_item_for_hash(self, file_hash: str, conn: sqlite3.Connection) -> NfoPlanItem | None:
        """Build a plan item for a single file hash."""
        # Get video info
        row = conn.execute(
            "SELECT file_hash, last_known_filepath FROM scanned_files WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()

        if not row:
            return None

        video_path = row["last_known_filepath"]
        file_name = os.path.basename(video_path) if video_path else None
        file_ext = os.path.splitext(video_path)[1] if video_path else None

        # Get people from DB (faces + video_people)
        db_people = self._get_db_people(file_hash, conn)

        # Find or determine NFO file path
        existing_nfo_path = self.nfo_service.find_nfo_path(video_path) if video_path else None
        # Use existing NFO or create new one at default location
        nfo_path = existing_nfo_path
        if video_path and not nfo_path and os.path.isfile(video_path):
            nfo_path = self.nfo_service.get_default_nfo_path(video_path)

        # Read existing actors from NFO
        existing_indexium_actors: list[str] = []
        other_actors: list[NfoActor] = []
        probe_error: str | None = None
        nfo_mtime: float | None = None

        if existing_nfo_path:
            try:
                nfo_mtime = os.path.getmtime(existing_nfo_path)
                all_actors = self.nfo_service.read_actors(existing_nfo_path)
                for actor in all_actors:
                    if actor.source == "indexium":
                        existing_indexium_actors.append(actor.name)
                    else:
                        other_actors.append(actor)
            except NfoParseError as e:
                probe_error = str(e)

        # Calculate changes
        db_set = set(db_people)
        existing_set = set(existing_indexium_actors)

        tags_to_add = sorted(db_set - existing_set, key=str.lower)
        tags_to_remove = sorted(existing_set - db_set, key=str.lower)
        result_people = sorted(db_set, key=str.lower)

        # Determine can_update: need a target nfo_path and no parse errors
        # (nfo_path will be set if video exists, even if NFO file doesn't yet exist)
        can_update = nfo_path is not None and probe_error is None
        risk_level = self._determine_risk_level(
            can_write=nfo_path is not None,
            probe_error=probe_error,
            other_actors=other_actors,
            db_people=db_people,
            tags_to_remove=tags_to_remove,
        )

        # Build comment strings for UI compatibility
        existing_comment = ", ".join(sorted(existing_indexium_actors, key=str.lower)) or None
        result_comment = ", ".join(result_people)

        # Non-indexium actors in NFO (for UI display)
        metadata_only_people = [a.name for a in other_actors]

        return NfoPlanItem(
            file_hash=file_hash,
            file_path=video_path,
            file_name=file_name,
            file_extension=file_ext,
            nfo_path=nfo_path,
            db_people=db_people,
            existing_people=list(existing_indexium_actors),
            result_people=result_people,
            tags_to_add=tags_to_add,
            tags_to_remove=tags_to_remove,
            existing_indexium_actors=list(existing_indexium_actors),
            other_actors=other_actors,
            existing_comment=existing_comment,
            result_comment=result_comment,
            risk_level=risk_level,
            can_update=can_update,
            probe_error=probe_error,
            metadata_only_people=metadata_only_people,
            will_overwrite_comment=bool(existing_indexium_actors and tags_to_remove),
            overwrites_custom_comment=risk_level == "danger",
            tag_count=len(result_people),
            new_tag_count=len(tags_to_add),
            file_modified_time=nfo_mtime,
        )

    def _get_db_people(self, file_hash: str, conn: sqlite3.Connection) -> list[str]:
        """Get all people tagged for this video in DB."""
        people = set()

        # From faces table
        rows = conn.execute(
            "SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL",
            (file_hash,),
        ).fetchall()
        for row in rows:
            people.add(row[0])

        # From video_people table (manual tagging)
        rows = conn.execute(
            "SELECT DISTINCT person_name FROM video_people WHERE file_hash = ?",
            (file_hash,),
        ).fetchall()
        for row in rows:
            people.add(row[0])

        return sorted(people, key=str.lower)

    def filter_items(
        self,
        items: list[NfoPlanItem],
        filters: dict[str, Any] | None = None,
    ) -> list[NfoPlanItem]:
        """Filter plan items based on criteria."""
        if not filters:
            return list(items)

        filtered = list(items)

        risk_levels = filters.get("risk_levels")
        if risk_levels:
            normalized = {level.lower() for level in risk_levels}

            def risk_match(plan_item: NfoPlanItem) -> bool:
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
            filtered = [item for item in filtered if item.tag_count >= min_tags]
        if max_tags is not None:
            filtered = [item for item in filtered if item.tag_count <= max_tags]

        file_types = filters.get("file_types")
        if file_types:
            # Normalize extensions (handle both "mp4" and ".mp4")
            normalized_types = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_types
            }
            filtered = [
                item
                for item in filtered
                if item.file_extension and item.file_extension.lower() in normalized_types
            ]

        # Search filter - matches file name, people, or file hash
        search = (filters.get("search") or "").strip().lower()
        if search:
            filtered = [
                item
                for item in filtered
                if search in (item.file_name or "").lower()
                or any(search in person.lower() for person in item.result_people)
                or search in (item.file_hash or "").lower()
            ]

        # Issue codes filter
        issue_codes = filters.get("issue_codes")
        if issue_codes:
            wanted = {code.lower() for code in issue_codes}
            filtered = [
                item
                for item in filtered
                if wanted.intersection(code.lower() for code in item.issue_codes)
            ]

        return filtered

    def sort_items(
        self,
        items: list[NfoPlanItem],
        sort_by: str | None = None,
        direction: str = "asc",
    ) -> list[NfoPlanItem]:
        """Sort plan items by specified field."""
        if not sort_by:
            return list(items)

        reverse = direction.lower() == "desc"

        def risk_priority(level: str) -> int:
            return {"danger": 3, "warning": 2, "safe": 1, "blocked": 0}.get(level, 0)

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
        if sort_by == "file_name":
            return sorted(
                items,
                key=lambda item: (item.file_name or item.file_hash).lower(),
                reverse=reverse,
            )
        # Alphabetical sort (maps to file_name sort)
        if sort_by == "alphabetical":
            return sorted(
                items,
                key=lambda item: (item.file_name or item.file_hash).lower(),
                reverse=reverse,
            )
        # Modified time sort
        if sort_by == "modified":
            return sorted(
                items,
                key=lambda item: (item.file_modified_time or 0.0, item.file_name or item.file_hash),
                reverse=reverse,
            )
        return list(items)

    def update_item_with_custom_people(
        self,
        item: NfoPlanItem,
        custom_people: list[str],
    ) -> NfoPlanItem:
        """Create updated item with custom people list."""
        result_people = sorted(set(p.strip() for p in custom_people if p.strip()), key=str.lower)
        if not result_people:
            result_people = list(item.result_people)

        existing_set = set(item.existing_indexium_actors)
        result_set = set(result_people)

        tags_to_add = sorted(result_set - existing_set, key=str.lower)
        tags_to_remove = sorted(existing_set - result_set, key=str.lower)

        issues = list(item.issues)
        issue_codes = list(item.issue_codes)
        risk_level = item.risk_level

        if tags_to_remove and risk_level == "safe":
            risk_level = "warning"
            if "tag_removal" not in issue_codes:
                issue_codes.append("tag_removal")
            removal_issue = "Removing actors from NFO"
            if removal_issue not in issues:
                issues.append(removal_issue)

        result_comment = ", ".join(result_people) if result_people else ""
        will_overwrite = bool(item.existing_indexium_actors and tags_to_remove)

        return NfoPlanItem(
            file_hash=item.file_hash,
            file_path=item.file_path,
            file_name=item.file_name,
            file_extension=item.file_extension,
            nfo_path=item.nfo_path,
            db_people=item.db_people,
            existing_people=item.existing_people,
            result_people=result_people,
            tags_to_add=tags_to_add,
            tags_to_remove=tags_to_remove,
            existing_indexium_actors=item.existing_indexium_actors,
            other_actors=item.other_actors,
            existing_comment=item.existing_comment,
            result_comment=result_comment,
            risk_level=risk_level,
            can_update=item.can_update,
            issues=issues,
            issue_codes=issue_codes,
            probe_error=item.probe_error,
            metadata_only_people=item.metadata_only_people,
            will_overwrite_comment=will_overwrite,
            overwrites_custom_comment=item.overwrites_custom_comment,
            tag_count=len(result_people),
            new_tag_count=len(tags_to_add),
            file_modified_time=item.file_modified_time,
        )

    def _determine_risk_level(
        self,
        can_write: bool,
        probe_error: str | None,
        other_actors: list[NfoActor],
        db_people: list[str],
        tags_to_remove: list[str],
    ) -> str:
        """Determine risk level for this item."""
        if not can_write:
            return "blocked"  # Can't write (no video file or no target path)

        if probe_error:
            return "blocked"  # XML parse error

        # Check for corrupted state: non-indexium actors that match DB names
        other_names = {a.name.lower() for a in other_actors}
        db_names = {n.lower() for n in db_people}
        if other_names & db_names:
            return "danger"  # Would modify non-indexium actors

        if tags_to_remove:
            return "warning"  # Removing previously tagged actors

        return "safe"  # Only adding new actors


@dataclass
class _NfoWriterRuntime:
    """In-memory control block for an active NFO write operation."""

    operation_id: int
    items: list[NfoPlanItem]
    item_ids: list[int]
    pause_event: threading.Event
    cancel_event: threading.Event
    written_nfos: set[str] = field(default_factory=set)
    thread: threading.Thread | None = None


class NfoWriter:
    """Service responsible for executing NFO writes asynchronously."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.nfo_service = NfoService()
        self.backup_manager = NfoBackupManager()
        self._lock = threading.Lock()
        self._operations: dict[int, _NfoWriterRuntime] = {}

    def start_operation(
        self,
        items: list[NfoPlanItem],
        backup: bool = True,
        background: bool = True,
    ) -> int:
        """Create an NFO write operation and dispatch background processing."""
        # Filter to only items that need updates
        runnable_items = [item for item in items if item.requires_update]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metadata_operations (operation_type, status, file_count)
                VALUES (?, ?, ?)
                """,
                ("nfo_write", "pending", len(items)),
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
                        nfo_path,
                        status,
                        previous_comment,
                        new_comment,
                        tags_added,
                        tags_removed
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        plan_item.file_hash,
                        plan_item.file_path or "",
                        plan_item.nfo_path,
                        "pending" if plan_item.requires_update else "skipped",
                        plan_item.existing_comment,
                        plan_item.result_comment,
                        ",".join(plan_item.tags_to_add),
                        ",".join(plan_item.tags_to_remove),
                    ),
                )
                item_id = cursor.lastrowid
                assert item_id is not None, "INSERT must return a lastrowid"
                item_ids.append(item_id)

            conn.commit()

        runnable_ids = [
            item_id for item, item_id in zip(items, item_ids, strict=False) if item.requires_update
        ]
        runtime = _NfoWriterRuntime(
            operation_id=operation_id,
            items=list(runnable_items),
            item_ids=runnable_ids,
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
        )
        runtime.pause_event.set()  # Start unpaused

        with self._lock:
            self._operations[operation_id] = runtime

        if background:
            thread = threading.Thread(
                target=self._run_operation,
                args=(runtime, backup),
                daemon=True,
                name=f"nfo-writer-{operation_id}",
            )
            runtime.thread = thread
            thread.start()
        else:
            self._run_operation(runtime, backup)

        return operation_id

    def get_operation_status(self, operation_id: int) -> dict[str, Any] | None:
        """Get current status of an NFO write operation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            op_row = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if not op_row:
                return None

            item_rows = conn.execute(
                """
                SELECT id, file_hash, file_path, nfo_path, status,
                       previous_comment, new_comment, tags_added, tags_removed,
                       error_message, processed_at
                FROM metadata_operation_items
                WHERE operation_id = ?
                ORDER BY id
                """,
                (operation_id,),
            ).fetchall()

        totals = {"success": 0, "failed": 0, "pending": 0, "skipped": 0}
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

            file_path = row["file_path"]
            file_name = os.path.basename(file_path) if file_path else row["file_hash"]
            tags_added_raw = row["tags_added"]
            tags_added = tags_added_raw.split(",") if tags_added_raw else []

            items.append(
                {
                    "id": row["id"],
                    "file_hash": row["file_hash"],
                    "file_path": file_path,
                    "file_name": file_name,
                    "nfo_path": row["nfo_path"],
                    "status": status,
                    "tags_added": tags_added,
                    "error_message": row["error_message"],
                    "processed_at": row["processed_at"],
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

    def pause_operation(self, operation_id: int) -> bool:
        """Pause an in-progress operation."""
        with self._lock:
            runtime = self._operations.get(operation_id)
        if not runtime:
            return False
        runtime.pause_event.clear()
        self._update_operation_status(operation_id, "paused")
        return True

    def resume_operation(self, operation_id: int) -> bool:
        """Resume a paused operation."""
        with self._lock:
            runtime = self._operations.get(operation_id)
        if not runtime:
            return False
        runtime.pause_event.set()
        self._update_operation_status(operation_id, "in_progress")
        return True

    def cancel_operation(self, operation_id: int) -> bool:
        """Cancel an operation."""
        with self._lock:
            runtime = self._operations.get(operation_id)
        if not runtime:
            return False
        runtime.cancel_event.set()
        runtime.pause_event.set()  # Unblock if paused
        self._update_operation_status(operation_id, "cancelling")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_operation(self, runtime: _NfoWriterRuntime, backup: bool) -> None:
        """Background worker for NFO write operation."""
        operation_id = runtime.operation_id
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE metadata_operations
                   SET status = 'in_progress', started_at = COALESCE(started_at, CURRENT_TIMESTAMP)
                   WHERE id = ?""",
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
                    self._write_single_item(plan_item, operation_id, backup, runtime)
                    self._record_success(operation_id, item_id)
                    processed_ids.add(item_id)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to write NFO for %s", plan_item.file_hash)
                    self._record_failure(operation_id, item_id, str(exc))
                    processed_ids.add(item_id)

                if runtime.cancel_event.is_set():
                    break

            if runtime.cancel_event.is_set():
                # Mark remaining items as skipped
                for pending_id in runtime.item_ids:
                    if pending_id not in processed_ids:
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(
                                """UPDATE metadata_operation_items
                                   SET status = 'skipped', processed_at = CURRENT_TIMESTAMP
                                   WHERE id = ?""",
                                (pending_id,),
                            )
                            conn.commit()
                self._update_operation_status(operation_id, "cancelled")
            else:
                self._update_operation_status(operation_id, "completed")
        finally:
            with self._lock:
                self._operations.pop(operation_id, None)

    def _write_single_item(
        self,
        item: NfoPlanItem,
        operation_id: int,
        backup: bool,
        runtime: _NfoWriterRuntime,
    ) -> None:
        """Write NFO for a single plan item."""
        if not item.nfo_path:
            raise ValueError(f"No NFO path for {item.file_hash}")

        # Coordinate writes to shared NFO files
        # Only create backup and write if we haven't already written this NFO
        if item.nfo_path in runtime.written_nfos:
            logger.debug("Skipping already-written NFO: %s", item.nfo_path)
            return

        if backup:
            self.backup_manager.create_backup(item.nfo_path, operation_id)

        self.nfo_service.write_actors(item.nfo_path, item.result_people)
        runtime.written_nfos.add(item.nfo_path)

    def _update_operation_status(self, operation_id: int, status: str) -> None:
        """Update operation status in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE metadata_operations
                   SET status = ?,
                       completed_at = CASE WHEN ? IN ('completed', 'cancelled') THEN CURRENT_TIMESTAMP ELSE completed_at END
                   WHERE id = ?""",
                (status, status, operation_id),
            )
            conn.commit()

    def _mark_item_status(self, item_id: int, status: str) -> None:
        """Mark individual item status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE metadata_operation_items SET status = ? WHERE id = ?",
                (status, item_id),
            )
            conn.commit()

    def _record_success(self, operation_id: int, item_id: int) -> None:
        """Record successful item write."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE metadata_operation_items
                   SET status = 'success', processed_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (item_id,),
            )
            conn.execute(
                """UPDATE metadata_operations
                   SET success_count = COALESCE(success_count, 0) + 1
                   WHERE id = ?""",
                (operation_id,),
            )
            conn.commit()

    def _record_failure(self, operation_id: int, item_id: int, error: str) -> None:
        """Record failed item write."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE metadata_operation_items
                   SET status = 'failed', error_message = ?, processed_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (error, item_id),
            )
            conn.execute(
                """UPDATE metadata_operations
                   SET failure_count = COALESCE(failure_count, 0) + 1
                   WHERE id = ?""",
                (operation_id,),
            )
            conn.commit()


class NfoHistoryService:
    """Service for querying operation history and performing rollbacks."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.backup_manager = NfoBackupManager()

    def list_operations(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return recent operations for history UI."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            where = ""
            params: list[Any] = []
            if status_filter:
                where = "WHERE status = ?"
                params.append(status_filter)

            # Get total count
            total = conn.execute(
                f"SELECT COUNT(*) FROM metadata_operations {where}",  # noqa: S608
                params,
            ).fetchone()[0]

            # Get operations
            rows = conn.execute(
                f"SELECT * FROM metadata_operations {where} "  # noqa: S608
                "ORDER BY started_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

            return [dict(row) for row in rows], total

    def get_operation_detail(self, operation_id: int) -> dict[str, Any] | None:
        """Get full operation details including items."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            op_row = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if not op_row:
                return None

            items = conn.execute(
                "SELECT * FROM metadata_operation_items WHERE operation_id = ?",
                (operation_id,),
            ).fetchall()

            return {
                "operation": dict(op_row),
                "items": [dict(item) for item in items],
            }

    def rollback_operation(self, operation_id: int) -> dict[str, Any]:
        """Rollback operation using operation-scoped backups."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Check operation exists and is rollback-able
            op = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()

            if not op:
                return {"success": False, "error": "Operation not found"}

            if op["status"] not in ("completed", "failed"):
                return {"success": False, "error": "Operation not in rollback-able state"}

            # Get items with nfo_path
            items = conn.execute(
                "SELECT * FROM metadata_operation_items WHERE operation_id = ? AND nfo_path IS NOT NULL",
                (operation_id,),
            ).fetchall()

            restored = 0
            failed = 0
            restored_nfos: set[str] = set()

            for item in items:
                nfo_path = item["nfo_path"]
                if not nfo_path or nfo_path in restored_nfos:
                    continue

                if self.backup_manager.restore_backup(nfo_path, operation_id):
                    self.backup_manager.cleanup_backup(nfo_path, operation_id)
                    restored += 1
                    restored_nfos.add(nfo_path)
                else:
                    failed += 1

            # Update operation status
            conn.execute(
                "UPDATE metadata_operations SET status = 'rolled_back' WHERE id = ?",
                (operation_id,),
            )
            conn.execute(
                "UPDATE metadata_operation_items SET status = 'rolled_back' WHERE operation_id = ?",
                (operation_id,),
            )
            conn.commit()

            return {
                "success": True,
                "restored": restored,
                "failed": failed,
            }
