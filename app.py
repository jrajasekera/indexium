from __future__ import annotations

import json
import math
import os
import random
import re
import secrets
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from config import Config
from nfo_services import NfoHistoryService, NfoPlanItem, NfoPlanner, NfoWriter
from text_utils import MIN_FRAGMENT_LENGTH, calculate_top_text_fragments

config = Config()

app = Flask(__name__)
# Flask needs a secret key to use flash messages
app.secret_key = config.SECRET_KEY

# NFO metadata services
nfo_planner = NfoPlanner(db_path=config.DATABASE_FILE)
nfo_writer = NfoWriter(db_path=config.DATABASE_FILE)
nfo_history = NfoHistoryService(db_path=config.DATABASE_FILE)

# --- DATABASE & HELPERS ---

_manual_warmup_executor: ThreadPoolExecutor | None = None
_manual_warmup_inflight: set[str] = set()
_manual_warmup_lock = threading.Lock()

if config.MANUAL_VIDEO_REVIEW_ENABLED and config.MANUAL_REVIEW_WARMUP_ENABLED:
    worker_count = max(int(config.MANUAL_REVIEW_WARMUP_WORKERS or 1), 1)
    _manual_warmup_executor = ThreadPoolExecutor(max_workers=worker_count)

_known_people_cache_lock = threading.Lock()
_known_people_cache: dict[str, Any] = {
    "timestamp": 0.0,
    "names": [],
    "prepared": [],
    "source": None,
}


def get_db_connection() -> sqlite3.Connection:
    """Gets a per-request database connection."""
    if "db" not in g:
        g.db = sqlite3.connect(config.DATABASE_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db_connection(exception: BaseException | None) -> None:
    """Closes the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


MANUAL_ACTIVE_STATUSES = {"pending", "in_progress"}
MANUAL_FINAL_STATUSES = {"done", "no_people"}
MANUAL_ALL_STATUSES = MANUAL_ACTIVE_STATUSES | MANUAL_FINAL_STATUSES | {"not_required"}
SAMPLE_MAX_WIDTH = 640
SAMPLE_MAX_HEIGHT = 360
MIN_TEXT_FRAGMENT_LENGTH = max(MIN_FRAGMENT_LENGTH, config.OCR_MIN_TEXT_LENGTH)


def _load_known_people_cache(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return cached known people data, refreshing it if stale."""
    now = time.monotonic()
    cache_ttl = max(float(config.MANUAL_KNOWN_PEOPLE_CACHE_SECONDS or 0), 0.0)
    db_source = str(config.DATABASE_FILE)

    with _known_people_cache_lock:
        cached_timestamp = float(_known_people_cache["timestamp"])
        cached_source = _known_people_cache.get("source")
        if (
            cached_timestamp
            and cache_ttl > 0
            and now - cached_timestamp < cache_ttl
            and cached_source == db_source
        ):
            return _known_people_cache

    names: set[str] = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL"
        ).fetchall()
    }
    names.update(
        row[0] for row in conn.execute("SELECT DISTINCT person_name FROM video_people").fetchall()
    )

    sorted_names: list[str] = sorted(names, key=lambda name: name.lower())
    prepared: list[tuple[str, str, list[str]]] = []
    for name in sorted_names:
        normalized = _normalize_match_text(name)
        if not normalized or normalized == "unknown":
            continue
        prepared.append((name, normalized, normalized.split()))

    refreshed: dict[str, Any] = {
        "timestamp": now,
        "names": sorted_names,
        "prepared": prepared,
        "source": db_source,
    }

    with _known_people_cache_lock:
        _known_people_cache.update(refreshed)

    return _known_people_cache


def _invalidate_known_people_cache() -> None:
    """Clear the cached known people data."""
    with _known_people_cache_lock:
        _known_people_cache["timestamp"] = 0.0
        _known_people_cache["names"] = []
        _known_people_cache["prepared"] = []
        _known_people_cache["source"] = None


def _schedule_manual_video_warmup(next_hash: str | None) -> None:
    """Start background preloading for the next manual-review video."""
    if not next_hash or _manual_warmup_executor is None:
        return

    with _manual_warmup_lock:
        if next_hash in _manual_warmup_inflight:
            return
        _manual_warmup_inflight.add(next_hash)

    def _enqueue():
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(config.DATABASE_FILE)
            conn.row_factory = sqlite3.Row
            _get_video_samples(conn, next_hash)
            _load_known_people_cache(conn)
        except Exception:  # pragma: no cover - defensive logging
            app.logger.exception("Failed preloading manual video %s", next_hash)
        finally:
            if conn is not None:
                conn.close()
            with _manual_warmup_lock:
                _manual_warmup_inflight.discard(next_hash)

    _manual_warmup_executor.submit(_enqueue)


def _schedule_manual_video_warmup_batch(next_hashes: list[str] | tuple[str, ...] | None) -> None:
    """Queue warmup jobs for multiple upcoming manual-review videos."""
    if not next_hashes:
        return
    for next_hash in next_hashes:
        _schedule_manual_video_warmup(next_hash)


def _manual_feature_guard() -> None:
    """Abort with 404 if manual video review is disabled."""
    if not config.MANUAL_VIDEO_REVIEW_ENABLED:
        abort(404)


def _manual_status_sort_key(status: str | None) -> int:
    order: dict[str | None, int] = {
        "pending": 0,
        "in_progress": 1,
        "done": 2,
        "no_people": 3,
        "not_required": 4,
        None: 5,
    }
    return order.get(status, 6)


def _resolve_sample_dir(file_hash: str) -> Path:
    """Return the filesystem directory for a video's sampled frames."""
    return Path(config.NO_FACE_SAMPLE_DIR) / file_hash


def _ensure_sample_seed(
    conn: sqlite3.Connection, file_hash: str, regenerate: bool = False
) -> int | None:
    """Get or create the deterministic sampling seed for a video."""
    row = conn.execute(
        "SELECT sample_seed FROM scanned_files WHERE file_hash = ?",
        (file_hash,),
    ).fetchone()
    if not row:
        return None
    seed: int | None = row["sample_seed"]
    if regenerate or seed is None:
        seed = secrets.randbits(32)
        conn.execute(
            "UPDATE scanned_files SET sample_seed = ? WHERE file_hash = ?",
            (seed, file_hash),
        )
        conn.commit()
    return seed


def _resize_frame_for_sample(frame: Any) -> Any:
    """Resize frames for display while keeping aspect ratio."""
    height, width = frame.shape[:2]
    if not height or not width:
        return frame
    scale = min(SAMPLE_MAX_WIDTH / width, SAMPLE_MAX_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _generate_video_samples(
    video_path: str, sample_dir: Path, sample_count: int, seed: int
) -> list[Path]:
    """Generate sample frames from a video and persist them to disk."""
    generated: list[Path] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        app.logger.warning("Could not open video %s for manual sampling", video_path)
        return generated

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            app.logger.warning(
                "Video %s reports no frame count; falling back to sequential sampling", video_path
            )
            collected = 0
            frame_index = 0
            while collected < sample_count:
                success, frame = cap.read()
                if not success:
                    break
                frame = _resize_frame_for_sample(frame)
                filename = f"frame_seq_{frame_index:06d}.jpg"
                target_path = sample_dir / filename
                if cv2.imwrite(str(target_path), frame):
                    generated.append(target_path)
                    collected += 1
                frame_index += 1
            return generated

        rng = random.Random(seed)
        target = min(sample_count, total_frames)
        if total_frames <= target:
            indices = list(range(total_frames))
        else:
            indices = sorted(rng.sample(range(total_frames), target))

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            frame = _resize_frame_for_sample(frame)
            filename = f"frame_{idx:06d}.jpg"
            target_path = sample_dir / filename
            if cv2.imwrite(str(target_path), frame):
                generated.append(target_path)
        return generated
    finally:
        cap.release()


def _get_video_samples(
    conn: sqlite3.Connection, file_hash: str, regenerate: bool = False
) -> tuple[list[Path], str | None]:
    """Ensure sample frames exist for a video and return them with the source path."""
    row = conn.execute(
        "SELECT last_known_filepath FROM scanned_files WHERE file_hash = ?",
        (file_hash,),
    ).fetchone()
    if not row:
        return [], None

    video_path: str | None = row["last_known_filepath"]
    if not video_path or not os.path.exists(video_path):
        return [], video_path

    sample_dir = _resolve_sample_dir(file_hash)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if regenerate:
        for existing in sample_dir.glob("*.jpg"):
            try:
                existing.unlink()
            except OSError:
                app.logger.warning("Failed to remove old sample %s", existing)

    existing_paths = sorted(sample_dir.glob("*.jpg"))
    sample_count = config.NO_FACE_SAMPLE_COUNT

    if (
        existing_paths
        and not regenerate
        and len(existing_paths) >= min(sample_count, len(existing_paths))
    ):
        return existing_paths[:sample_count], video_path

    seed = _ensure_sample_seed(conn, file_hash, regenerate or not existing_paths)
    if seed is None:
        return existing_paths[:sample_count], video_path

    generated = _generate_video_samples(video_path, sample_dir, sample_count, seed)
    if not generated and existing_paths:
        return existing_paths[:sample_count], video_path
    return generated, video_path


def _collect_known_people(conn: sqlite3.Connection) -> list[str]:
    """Return a sorted list of known people names from cache or database."""
    cache = _load_known_people_cache(conn)
    return list(cache["names"])


def _normalize_match_text(text: str) -> str:
    """Normalize text for loose matching (lowercase, alphanumeric, single spaces)."""
    lowered = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _candidate_windows(tokens: list[str], target_length: int) -> list[str]:
    """Return candidate substrings for matching against known names."""
    if not tokens:
        return []
    windows: set[str] = {" ".join(tokens)}
    if target_length <= 0:
        return list(windows)
    if target_length <= len(tokens):
        for idx in range(len(tokens) - target_length + 1):
            windows.add(" ".join(tokens[idx : idx + target_length]))
    return list(windows)


def _suggest_manual_person_name(
    prepared_people: list[tuple[str, str, list[str]]],
    candidate_texts: list[tuple[str, str]],
    excluded_names: set[str],
) -> tuple[str | None, float | None, str | None]:
    """Return the best matching known person based on filename/OCR content."""

    prepared_candidates: list[tuple[str, str, list[str]]] = []
    seen_norms: set[str] = set()
    for source, raw_text in candidate_texts:
        normalized = _normalize_match_text(raw_text)
        if not normalized or normalized in seen_norms:
            continue
        seen_norms.add(normalized)
        prepared_candidates.append((source, normalized, normalized.split()))

    if not prepared_candidates or not prepared_people:
        return None, None, None

    excluded_normalized = {_normalize_match_text(name) for name in excluded_names}

    best_name: str | None = None
    best_score: float = 0.0
    best_source: str | None = None

    for name, normalized_name, name_tokens in prepared_people:
        if (
            not normalized_name
            or normalized_name == "unknown"
            or normalized_name in excluded_normalized
        ):
            continue

        expected_length = len(name_tokens)
        spaceless_name = normalized_name.replace(" ", "")

        for source, _candidate_norm, tokens in prepared_candidates:
            # Check if concatenated name appears as substring (e.g., "timothyjones" matches "Timothy Jones")
            spaceless_candidate = _candidate_norm.replace(" ", "")
            if spaceless_name in spaceless_candidate:
                return name, 0.95, source

            windows = _candidate_windows(tokens, expected_length)
            for window in windows:
                if not window:
                    continue
                if window == normalized_name:
                    return name, 1.0, source

                score = SequenceMatcher(None, normalized_name, window).ratio()
                if score > best_score:
                    best_name = name
                    best_score = score
                    best_source = source

    if best_name and best_score >= config.MANUAL_NAME_SUGGEST_THRESHOLD:
        return best_name, best_score, best_source

    return None, None, None


def _get_manual_video_record(conn: sqlite3.Connection, file_hash: str) -> sqlite3.Row:
    """Fetch a manual-review candidate row or abort if missing."""
    row = conn.execute(
        """
        SELECT file_hash,
               last_known_filepath,
               manual_review_status,
               face_count,
               sample_seed,
               last_attempt,
               ocr_text_count,
               ocr_last_updated
        FROM scanned_files
        WHERE file_hash = ?
        """,
        (file_hash,),
    ).fetchone()
    if not row:
        abort(404)
    return row


def _get_next_manual_video_hashes(
    conn: sqlite3.Connection, exclude_hash: str | None = None, limit: int = 1
) -> list[str]:
    """Return the next manual-review video hashes, optionally skipping current."""
    if limit <= 0:
        return []
    query = [
        """
        SELECT file_hash
        FROM scanned_files
        WHERE manual_review_status IN ('pending', 'in_progress')
        """
    ]
    params: list[Any] = []
    if exclude_hash:
        query.append("AND file_hash != ?")
        params.append(exclude_hash)
    query.append(
        "ORDER BY CASE manual_review_status WHEN 'pending' THEN 0 ELSE 1 END,"
        "         COALESCE(last_attempt, CURRENT_TIMESTAMP) ASC,"
        "         file_hash ASC"
    )
    query.append("LIMIT ?")
    params.append(int(limit))
    rows = conn.execute("\n".join(query), params).fetchall()
    return [row["file_hash"] for row in rows]


def _get_next_manual_video(conn: sqlite3.Connection, exclude_hash: str | None = None) -> str | None:
    """Return the next manual-review video hash, optionally skipping current."""
    hashes = _get_next_manual_video_hashes(conn, exclude_hash=exclude_hash, limit=1)
    return hashes[0] if hashes else None


def _serialize_plan_item(item: NfoPlanItem) -> dict[str, Any]:
    """Convert a plan item to the dict shape expected by the planner UI."""
    data = item.to_dict()
    data["name"] = item.file_name
    data["path"] = item.file_path
    return data


def build_metadata_plan(
    conn: sqlite3.Connection, target_hashes: list[str] | None = None
) -> list[dict[str, Any]]:
    """Builds a per-file plan summarizing pending metadata writes."""
    # Get all file hashes if not specified
    if target_hashes is None:
        rows = conn.execute(
            "SELECT file_hash FROM scanned_files WHERE processing_status = 'completed'"
        ).fetchall()
        target_hashes = [row["file_hash"] for row in rows]

    items = nfo_planner.build_plan(target_hashes)
    plan_dicts = [_serialize_plan_item(item) for item in items]
    plan_dicts.sort(key=lambda entry: ((entry["name"] or "").lower(), entry["file_hash"]))
    return plan_dicts


@app.route("/api/metadata/plan", methods=["GET", "POST"])
def api_get_metadata_plan():
    """Return a JSON representation of the metadata plan for the requested files."""
    conn = get_db_connection()
    payload = request.get_json(silent=True) or {}

    query_hashes = request.args.getlist("file_hashes")
    body_hashes = payload.get("file_hashes")
    if isinstance(body_hashes, str):
        body_hashes = [body_hashes]
    if body_hashes:
        query_hashes.extend(body_hashes)

    # Get all file hashes if not specified
    if query_hashes:
        file_hashes = query_hashes
    else:
        rows = conn.execute(
            "SELECT file_hash FROM scanned_files WHERE processing_status = 'completed'"
        ).fetchall()
        file_hashes = [row["file_hash"] for row in rows]

    filters = payload.get("filter") or {}

    raw_page = payload.get("page", request.args.get("page", 1))
    raw_per_page = payload.get("per_page", request.args.get("per_page", 50))
    try:
        page = max(1, int(raw_page))
    except (TypeError, ValueError):
        page = 1
    try:
        per_page = max(1, int(raw_per_page))
    except (TypeError, ValueError):
        per_page = 50

    sort_payload = payload.get("sort") or {}
    sort_by = sort_payload.get("by") or request.args.get("sort")
    sort_dir = sort_payload.get("direction") or request.args.get("direction", "asc")

    all_items = nfo_planner.build_plan(file_hashes)
    # Only show files that actually need updates (have a diff)
    items = [item for item in all_items if item.requires_update]
    filtered_items = nfo_planner.filter_items(items, filters)
    sorted_items = nfo_planner.sort_items(filtered_items, sort_by=sort_by, direction=sort_dir)
    total_items = len(filtered_items)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = sorted_items[start:end]

    # Compute statistics (items already filtered to only those requiring updates)
    total_files = len(items)
    safe_count = sum(1 for item in items if item.risk_level == "safe" and item.can_update)
    warning_count = sum(1 for item in items if item.risk_level == "warning" and item.can_update)
    danger_count = sum(1 for item in items if item.risk_level == "danger" and item.can_update)
    blocked_count = sum(1 for item in items if not item.can_update)

    # Compute categories inline
    categories = {
        "safe": [item.file_hash for item in items if item.risk_level == "safe" and item.can_update],
        "warning": [
            item.file_hash for item in items if item.risk_level == "warning" and item.can_update
        ],
        "danger": [
            item.file_hash for item in items if item.risk_level == "danger" and item.can_update
        ],
        "blocked": [item.file_hash for item in items if not item.can_update],
    }

    response = {
        "items": [_serialize_plan_item(item) for item in paginated_items],
        "statistics": {
            "total_files": total_files,
            "safe_count": safe_count,
            "warning_count": warning_count,
            "danger_count": danger_count,
            "blocked_count": blocked_count,
        },
        "categories": categories,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_items": total_items,
            "total_pages": max(1, math.ceil(total_items / per_page)) if per_page else 1,
        },
        "insights": {
            "total_people": sum(len(item.result_people) for item in items),
            "total_new_tags": sum(len(item.tags_to_add) for item in items),
            "blocked_files": blocked_count,
        },
        "filters": {
            "file_types": sorted({item.file_extension for item in items if item.file_extension}),
            "issue_codes": sorted({code for item in items for code in item.issue_codes}),
        },
        "sort": {
            "applied": {
                "by": sort_by or "alphabetical",
                "direction": sort_dir.lower(),
            }
        },
    }
    return jsonify(response)


@app.route("/api/metadata/plan/<file_hash>/edit", methods=["POST"])
def api_edit_metadata_plan_item(file_hash):
    """Simulate editing a plan item by returning an updated representation."""
    payload = request.get_json(force=True, silent=False) or {}
    result_people = payload.get("result_people", [])
    if isinstance(result_people, str):
        result_people = [result_people]
    if isinstance(result_people, tuple):
        result_people = list(result_people)
    if not isinstance(result_people, list):
        abort(400, description="result_people must be a list of names")

    normalized_people: list[str] = []
    for name in result_people:
        if not isinstance(name, str):
            abort(400, description="result_people must be a list of names")
        normalized_people.append(name)

    items = nfo_planner.build_plan([file_hash])
    if not items:
        abort(404, description="Plan item not found")

    updated_item = nfo_planner.update_item_with_custom_people(items[0], normalized_people)
    return jsonify({"item": updated_item.to_dict()})


@app.route("/metadata_progress")
def metadata_progress():
    """Render the async metadata write progress dashboard."""
    try:
        operation_id = int(request.args.get("operation_id", ""))
    except ValueError:
        operation_id = None

    if not operation_id:
        flash("Metadata operation not found. Start a new write from the planner.", "warning")
        return redirect(url_for("metadata_preview"))

    status = nfo_writer.get_operation_status(operation_id)
    if status is None:
        flash("Metadata operation not found or has already been cleaned up.", "warning")
        return redirect(url_for("metadata_preview"))

    return render_template("metadata_progress.html", operation_id=operation_id)


@app.route("/api/metadata/operations/<int:operation_id>", methods=["GET"])
def api_get_operation_status(operation_id: int):
    status = nfo_writer.get_operation_status(operation_id)
    if status is None:
        abort(404, description="Operation not found")
    return jsonify(status)


@app.route("/api/metadata/operations/<int:operation_id>/pause", methods=["POST"])
def api_pause_operation(operation_id: int):
    if nfo_writer.pause_operation(operation_id):
        return jsonify({"status": "paused"})
    abort(404, description="Operation not found or already finished")


@app.route("/api/metadata/operations/<int:operation_id>/resume", methods=["POST"])
def api_resume_operation(operation_id: int):
    if nfo_writer.resume_operation(operation_id):
        return jsonify({"status": "in_progress"})
    abort(404, description="Operation not found or already finished")


@app.route("/api/metadata/operations/<int:operation_id>/cancel", methods=["POST"])
def api_cancel_operation(operation_id: int):
    if nfo_writer.cancel_operation(operation_id):
        return jsonify({"status": "cancelling"})
    abort(404, description="Operation not found or already finished")


@app.route("/metadata_history")
def metadata_history():
    """Displays metadata operation history and rollback controls."""
    return render_template("metadata_history.html")


@app.route("/api/metadata/history", methods=["GET"])
def api_get_metadata_history():
    try:
        page = max(1, int(request.args.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        per_page = max(1, int(request.args.get("per_page", 20)))
    except (TypeError, ValueError):
        per_page = 20

    status_filter = request.args.get("status")
    offset = (page - 1) * per_page
    operations, total = nfo_history.list_operations(
        limit=per_page, offset=offset, status_filter=status_filter
    )

    return jsonify(
        {
            "operations": operations,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": max(1, math.ceil(total / per_page)) if per_page else 1,
            },
        }
    )


@app.route("/api/metadata/history/<int:operation_id>", methods=["GET"])
def api_get_metadata_history_details(operation_id: int):
    details = nfo_history.get_operation_detail(operation_id)
    if details is None:
        abort(404, description="Operation not found")
    return jsonify(details)


@app.route("/api/metadata/history/<int:operation_id>/rollback", methods=["POST"])
def api_rollback_metadata_operation(operation_id: int):
    result = nfo_history.rollback_operation(operation_id)
    if not result.get("success"):
        abort(400, description=result.get("error", "Rollback failed"))
    return jsonify(result)


def get_progress_stats():
    """Gets counts for UI display."""
    conn = get_db_connection()
    unnamed_groups_count = conn.execute("""
        SELECT COUNT(DISTINCT cluster_id) as count
        FROM faces
        WHERE cluster_id IS NOT NULL AND person_name IS NULL
    """).fetchone()["count"]
    named_people_count = conn.execute("""
        SELECT COUNT(DISTINCT person_name) as count
        FROM faces
        WHERE person_name IS NOT NULL
    """).fetchone()["count"]
    manual_pending_count = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM scanned_files
        WHERE manual_review_status IN ('pending', 'in_progress')
        """
    ).fetchone()["count"]
    manual_completed_count = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM scanned_files
        WHERE manual_review_status IN ('done', 'no_people')
        """
    ).fetchone()["count"]
    return {
        "unnamed_groups_count": unnamed_groups_count,
        "named_people_count": named_people_count,
        "manual_pending_videos": manual_pending_count,
        "manual_completed_videos": manual_completed_count,
    }


# Make stats available to all templates
@app.context_processor
def inject_stats():
    return dict(stats=get_progress_stats())


# --- ROUTES ---


@app.route("/")
def index():
    """Finds the next unnamed group and redirects to the tagging page for it."""
    conn = get_db_connection()
    skipped = session.get("skipped_clusters", [])

    base_query = """
        SELECT MIN(cluster_id) as id
        FROM faces
        WHERE cluster_id IS NOT NULL
          AND person_name IS NULL
          AND cluster_id != ?
    """

    params = []
    params.append(-1)
    if skipped:
        placeholders = ",".join("?" for _ in skipped)
        query = base_query + f" AND cluster_id NOT IN ({placeholders})"
        params.extend(skipped)
    else:
        query = base_query

    next_group = conn.execute(query, params).fetchone()

    if (not next_group or next_group["id"] is None) and skipped:
        session.pop("skipped_clusters", None)
        next_group = conn.execute(base_query, (-1,)).fetchone()

    if next_group and next_group["id"] is not None:
        return redirect(url_for("tag_group", cluster_id=next_group["id"]))
    else:
        # No more groups to name, show a completion page
        return render_template("all_done.html")


@app.route("/group/<int:cluster_id>")
def tag_group(cluster_id):
    """Displays a single group for tagging."""
    conn = get_db_connection()
    page = max(1, int(request.args.get("page", 1)))
    PAGE_SIZE = 50

    total_faces = conn.execute(
        "SELECT COUNT(*) as count FROM faces WHERE cluster_id = ?",
        (cluster_id,),
    ).fetchone()["count"]
    total_pages = max(1, math.ceil(total_faces / PAGE_SIZE))
    page = min(page, total_pages)
    offset = (page - 1) * PAGE_SIZE

    sample_faces = conn.execute(
        "SELECT id, suggested_person_name, suggested_confidence, suggestion_status, suggested_candidates FROM faces "
        "WHERE cluster_id = ? LIMIT ? OFFSET ?",
        (cluster_id, PAGE_SIZE, offset),
    ).fetchall()

    file_rows = conn.execute(
        """
        SELECT DISTINCT sf.last_known_filepath, sf.file_hash
        FROM faces f
        JOIN scanned_files sf ON f.file_hash = sf.file_hash
        WHERE f.cluster_id = ?
    """,
        (cluster_id,),
    ).fetchall()
    file_names = [os.path.basename(row["last_known_filepath"]) for row in file_rows]
    file_hashes = [row["file_hash"] for row in file_rows]
    files_data = list(zip(file_names, file_hashes, strict=False))
    file_name_by_hash = {file_hash: name for name, file_hash in files_data}

    ocr_aggregated = []
    ocr_by_file = {}
    ocr_top_fragments = []
    if file_hashes:
        placeholders = ",".join("?" for _ in file_hashes)
        rows = conn.execute(
            f"""
                SELECT vt.file_hash, vt.raw_text, vt.occurrence_count
                FROM video_text vt
                WHERE vt.file_hash IN ({placeholders})
            """,
            file_hashes,
        ).fetchall()

        dedup_global = {}
        for row in rows:
            raw_text = (row["raw_text"] or "").strip()
            if not raw_text or len(raw_text) < MIN_TEXT_FRAGMENT_LENGTH:
                continue
            normalized = raw_text.lower()
            per_file = ocr_by_file.setdefault(row["file_hash"], [])
            if all(text.lower() != normalized for text in per_file):
                per_file.append(raw_text)

            entry = dedup_global.setdefault(
                normalized, {"raw_text": raw_text, "file_hashes": set()}
            )
            entry["raw_text"] = raw_text
            entry["file_hashes"].add(row["file_hash"])

        for _file_hash, values in ocr_by_file.items():
            values.sort(key=lambda text: text.lower())

        ocr_aggregated = []
        for data in dedup_global.values():
            file_hash_list = sorted(data["file_hashes"])
            ocr_aggregated.append(
                {
                    "raw_text": data["raw_text"],
                    "file_hashes": file_hash_list,
                    "file_count": len(file_hash_list),
                    "file_names": [file_name_by_hash.get(fh, fh) for fh in file_hash_list],
                }
            )
        ocr_aggregated.sort(key=lambda item: item["raw_text"].lower())

        fragment_rows = conn.execute(
            f"""
                SELECT file_hash, fragment_text, occurrence_count, text_length, rank
                FROM video_text_fragments
                WHERE file_hash IN ({placeholders})
                ORDER BY rank
            """,
            file_hashes,
        ).fetchall()

        if fragment_rows:
            fragment_entries = [
                (row["fragment_text"], row["occurrence_count"]) for row in fragment_rows
            ]
            ocr_top_fragments = calculate_top_text_fragments(
                fragment_entries,
                top_n=5,
                min_length=MIN_TEXT_FRAGMENT_LENGTH,
            )

    ocr_text_by_file_items = [
        {
            "file_hash": file_hash,
            "file_name": file_name_by_hash.get(file_hash, file_hash),
            "texts": texts,
        }
        for file_hash, texts in ocr_by_file.items()
    ]
    ocr_text_by_file_items.sort(key=lambda item: item["file_name"].lower())

    if not sample_faces:
        flash(f"Cluster #{cluster_id} no longer exists or is empty.", "error")
        return redirect(url_for("index"))

    names = conn.execute(
        "SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL ORDER BY person_name"
    ).fetchall()

    existing_names = [name["person_name"] for name in names]

    suggestion_rows = conn.execute(
        "SELECT suggested_candidates FROM faces WHERE cluster_id = ? AND suggested_candidates IS NOT NULL",
        (cluster_id,),
    ).fetchall()

    candidate_summary = {}
    for row in suggestion_rows:
        raw = row[0]
        if not raw:
            continue
        try:
            candidates = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            candidates = []
        for candidate in candidates:
            name = candidate.get("name")
            confidence = candidate.get("confidence")
            if not name or confidence is None:
                continue
            bucket = candidate_summary.setdefault(
                name,
                {
                    "name": name,
                    "total_confidence": 0.0,
                    "count": 0,
                    "max_confidence": 0.0,
                },
            )
            bucket["total_confidence"] += confidence
            bucket["count"] += 1
            bucket["max_confidence"] = max(bucket["max_confidence"], confidence)

    suggestion_candidates = []
    for bucket in candidate_summary.values():
        avg_confidence = bucket["total_confidence"] / bucket["count"]
        suggestion_candidates.append(
            {
                "name": bucket["name"],
                "avg_confidence": avg_confidence,
                "max_confidence": bucket["max_confidence"],
                "count": bucket["count"],
            }
        )

    suggestion_candidates.sort(
        key=lambda item: (item["avg_confidence"], item["max_confidence"], item["count"]),
        reverse=True,
    )
    suggestion_candidates = suggestion_candidates[:5]
    primary_suggestion = suggestion_candidates[0] if suggestion_candidates else None

    cluster_data = {
        "id": cluster_id,
        "faces": sample_faces,
        "page": page,
        "total_pages": total_pages,
    }
    return render_template(
        "group_tagger.html",
        cluster=cluster_data,
        existing_names=existing_names,
        file_names=file_names,
        file_hashes=file_hashes,
        files_data=files_data,
        file_name_by_hash=file_name_by_hash,
        ocr_text_by_file=ocr_text_by_file_items,
        ocr_text_aggregated=ocr_aggregated,
        ocr_top_fragments=ocr_top_fragments,
        suggestion_candidates=suggestion_candidates,
        primary_suggestion=primary_suggestion,
        cluster_is_unknown=(cluster_id == -1),
    )


@app.route("/face_thumbnail/<int:face_id>")
def get_face_thumbnail(face_id):
    """Serves a pre-generated face thumbnail."""
    thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{face_id}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="image/jpeg")
    return "Thumbnail not found", 404


# --- MANUAL VIDEO TAGGING ROUTES ---


@app.route("/videos/manual")
def manual_video_dashboard():
    """Lists videos that need manual people tagging."""
    _manual_feature_guard()
    conn = get_db_connection()
    focus_person = request.args.get("person", "").strip()
    focus_person_normalized = focus_person.lower() if focus_person else None
    rows = conn.execute(
        """
        SELECT
            sf.file_hash,
            sf.last_known_filepath,
            sf.manual_review_status,
            sf.face_count,
            sf.last_attempt,
            (SELECT COUNT(*) FROM video_people vp WHERE vp.file_hash = sf.file_hash) AS tag_count
        FROM scanned_files sf
        WHERE sf.manual_review_status != 'not_required'
        ORDER BY
            CASE sf.manual_review_status
                WHEN 'pending' THEN 0
                WHEN 'in_progress' THEN 1
                WHEN 'done' THEN 2
                WHEN 'no_people' THEN 3
                ELSE 4
            END,
            sf.last_attempt DESC
        """
    ).fetchall()

    tag_rows = conn.execute("SELECT file_hash, person_name FROM video_people").fetchall()
    tags_by_file = {}
    for row in tag_rows:
        tags_by_file.setdefault(row["file_hash"], []).append(row["person_name"])

    videos = []
    for row in rows:
        video_path = row["last_known_filepath"]
        tags = sorted(tags_by_file.get(row["file_hash"], []), key=lambda name: name.lower())
        matches_focus = bool(focus_person_normalized) and any(
            tag.lower() == focus_person_normalized for tag in tags
        )
        needs_focus = (
            bool(focus_person_normalized)
            and not matches_focus
            and row["manual_review_status"] in MANUAL_ACTIVE_STATUSES
        )
        videos.append(
            {
                "file_hash": row["file_hash"],
                "name": os.path.basename(video_path) if video_path else None,
                "path": video_path,
                "status": row["manual_review_status"],
                "tag_count": row["tag_count"],
                "face_count": row["face_count"],
                "missing": not (video_path and os.path.exists(video_path)),
                "last_attempt": row["last_attempt"],
                "tags": tags,
                "matches_focus": matches_focus,
                "needs_focus": needs_focus,
            }
        )

    counts = {status: 0 for status in MANUAL_ALL_STATUSES}
    for video in videos:
        counts[video["status"]] = counts.get(video["status"], 0) + 1
    counts["total"] = len(videos)

    def status_order(status):
        return {
            "pending": 0,
            "in_progress": 1,
            "done": 2,
            "no_people": 3,
            "not_required": 4,
        }.get(status, 5)

    next_video = next(
        (video for video in videos if video["status"] in MANUAL_ACTIVE_STATUSES), None
    )

    if focus_person:
        matching_videos = [video for video in videos if video["matches_focus"]]
        suggested_videos = [video for video in videos if video["needs_focus"]]
        other_videos = [
            video
            for video in videos
            if video not in matching_videos and video not in suggested_videos
        ]
    else:
        matching_videos = []
        suggested_videos = []
        other_videos = videos

    videos.sort(
        key=lambda video: (
            status_order(video["status"]),
            (video["name"] or "").lower(),
            video["file_hash"],
        )
    )

    matching_videos.sort(
        key=lambda video: (
            status_order(video["status"]),
            (video["name"] or "").lower(),
            video["file_hash"],
        )
    )
    suggested_videos.sort(
        key=lambda video: (
            status_order(video["status"]),
            (video["name"] or "").lower(),
            video["file_hash"],
        )
    )
    other_videos.sort(
        key=lambda video: (
            status_order(video["status"]),
            (video["name"] or "").lower(),
            video["file_hash"],
        )
    )

    known_people_cache = _load_known_people_cache(conn)
    known_people = list(known_people_cache["names"])
    list(known_people_cache["prepared"])

    return render_template(
        "video_manual_list.html",
        videos=videos,
        counts=counts,
        next_video=next_video,
        focus_person=focus_person,
        matching_videos=matching_videos,
        suggested_videos=suggested_videos,
        other_videos=other_videos,
        known_people=known_people,
    )


@app.route("/videos/manual/next")
def manual_video_next():
    """Redirects to the next video requiring manual tagging."""
    _manual_feature_guard()
    conn = get_db_connection()
    exclude_hash = request.args.get("exclude")
    next_hash = _get_next_manual_video(conn, exclude_hash=exclude_hash)
    if not next_hash:
        flash("No videos waiting for manual tagging.", "info")
        return redirect(url_for("manual_video_dashboard"))
    return redirect(url_for("manual_video_detail", file_hash=next_hash))


@app.route("/videos/manual/<file_hash>")
def manual_video_detail(file_hash):
    """Displays sampling frames and tagging controls for a single video."""
    _manual_feature_guard()
    conn = get_db_connection()
    record = _get_manual_video_record(conn, file_hash)
    focus_person = request.args.get("focus_person", "").strip()

    status = record["manual_review_status"]
    if status == "not_required":
        flash("Video no longer requires manual tagging.", "info")
        return redirect(url_for("manual_video_dashboard"))
    if status == "pending":
        conn.execute(
            "UPDATE scanned_files SET manual_review_status = 'in_progress' WHERE file_hash = ?",
            (file_hash,),
        )
        conn.commit()
        status = "in_progress"

    sample_paths, video_path = _get_video_samples(conn, file_hash)
    sample_files = [
        {
            "filename": path.name,
            "url": url_for("manual_video_frame", file_hash=file_hash, filename=path.name),
        }
        for path in sample_paths
    ]

    if not video_path:
        video_path = record["last_known_filepath"]

    tags = [
        row["person_name"]
        for row in conn.execute(
            "SELECT person_name FROM video_people WHERE file_hash = ? ORDER BY LOWER(person_name)",
            (file_hash,),
        ).fetchall()
    ]

    text_rows = conn.execute(
        """
        SELECT raw_text, confidence, occurrence_count
        FROM video_text
        WHERE file_hash = ?
        ORDER BY LOWER(raw_text)
        """,
        (file_hash,),
    ).fetchall()

    ocr_entries = []
    seen_texts = set()
    for row in text_rows:
        raw_text = (row["raw_text"] or "").strip()
        if not raw_text or len(raw_text) < MIN_TEXT_FRAGMENT_LENGTH:
            continue
        normalized = raw_text.lower()
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        occurrence = row["occurrence_count"] or 1
        ocr_entries.append(
            {
                "raw_text": raw_text,
                "confidence": row["confidence"],
                "occurrence_count": occurrence,
            }
        )

    fragment_rows = conn.execute(
        """
        SELECT fragment_text, occurrence_count, text_length
        FROM video_text_fragments
        WHERE file_hash = ?
        ORDER BY rank
        LIMIT 5
        """,
        (file_hash,),
    ).fetchall()

    video_top_fragments = [
        {
            "substring": row["fragment_text"],
            "count": row["occurrence_count"],
            "length": row["text_length"],
        }
        for row in fragment_rows
    ]

    known_people_cache = _load_known_people_cache(conn)
    known_people = list(known_people_cache["names"])
    prepared_people = list(known_people_cache["prepared"])
    video_info = {
        "file_hash": file_hash,
        "name": os.path.basename(video_path) if video_path else None,
        "path": video_path,
        "status": status,
        "face_count": record["face_count"],
        "tag_count": len(tags),
        "ocr_text_count": record["ocr_text_count"] or 0,
        "ocr_last_updated": record["ocr_last_updated"],
        "missing": not (video_path and os.path.exists(video_path)),
    }

    candidate_texts: list[tuple[str, str]] = []
    if video_path:
        path_obj = Path(video_path)
        candidate_texts.append(("filename", path_obj.stem))
        candidate_texts.append(("filename", path_obj.name))

    for entry in ocr_entries:
        candidate_texts.append(("ocr_text", entry["raw_text"]))

    for fragment in video_top_fragments:
        candidate_texts.append(("ocr_fragment", fragment["substring"]))

    suggestion_name, suggestion_score, suggestion_source = _suggest_manual_person_name(
        prepared_people,
        candidate_texts,
        excluded_names=set(tags),
    )

    warmup_depth = max(int(config.MANUAL_REVIEW_WARMUP_DEPTH or 0), 0)
    if warmup_depth > 0:
        next_manual_hashes = _get_next_manual_video_hashes(
            conn, exclude_hash=file_hash, limit=warmup_depth
        )
        _schedule_manual_video_warmup_batch(next_manual_hashes)

    if suggestion_name:
        app.logger.info(
            "Auto-suggested '%s' for manual video %s via %s (score %.2f)",
            suggestion_name,
            file_hash,
            suggestion_source or "fuzzy",
            suggestion_score or 1.0,
        )

    return render_template(
        "video_manual_detail.html",
        video=video_info,
        sample_images=sample_files,
        tags=tags,
        known_people=known_people,
        sample_count=len(sample_files),
        focus_person=focus_person,
        video_text_entries=ocr_entries,
        video_text_top_fragments=video_top_fragments,
        auto_suggested_person=suggestion_name,
        auto_suggested_source=suggestion_source,
    )


@app.route("/videos/manual/<file_hash>/tags", methods=["POST"])
def manual_video_add_tags(file_hash):
    """Adds a single tag to a manual video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)

    focus_person = request.form.get("focus_person", "").strip()
    submit_action = request.form.get("submit_action", "add")
    submitted_name = request.form.get("person_name", "").strip()

    if not submitted_name:
        flash("Provide a name to add.", "warning")
        redirect_args = {"file_hash": file_hash}
        if focus_person:
            redirect_args["focus_person"] = focus_person
        return redirect(url_for("manual_video_detail", **redirect_args))

    added: list[str] = []
    duplicates: list[str] = []
    invalid: list[str] = []
    should_commit = False

    name = submitted_name
    if name.lower() == "unknown":
        invalid.append(name)
        should_commit = True
    else:
        try:
            conn.execute(
                "INSERT INTO video_people (file_hash, person_name) VALUES (?, ?)",
                (file_hash, name),
            )
            added.append(name)
            should_commit = True
        except sqlite3.IntegrityError:
            duplicates.append(name)
            should_commit = True

    if added:
        flash(f"Tagged video with: {', '.join(added)}", "success")
    if duplicates:
        flash(f"Already tagged: {', '.join(duplicates)}", "info")
    if invalid:
        flash("'Unknown' is reserved. Use the review buttons instead.", "error")

    mark_done_redirect: str | None = None
    next_manual_hash: str | None = None
    next_manual_hashes: list[str] = []
    warmup_depth = max(int(config.MANUAL_REVIEW_WARMUP_DEPTH or 0), 0)
    mark_done_requested = submit_action == "add_and_done"

    if mark_done_requested and not invalid:
        tag_count = conn.execute(
            "SELECT COUNT(*) as count FROM video_people WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()["count"]
        if tag_count == 0:
            flash(
                'Add at least one person before marking as done, or choose "No people".', "warning"
            )
        else:
            conn.execute(
                "UPDATE scanned_files SET manual_review_status = ? WHERE file_hash = ?",
                ("done", file_hash),
            )
            should_commit = True
            flash("Marked video as manually tagged.", "success")

            next_manual_hashes = _get_next_manual_video_hashes(
                conn, exclude_hash=file_hash, limit=max(warmup_depth, 1)
            )
            next_manual_hash = next_manual_hashes[0] if next_manual_hashes else None
            if next_manual_hash:
                redirect_args = {"file_hash": next_manual_hash}
                if focus_person:
                    redirect_args["focus_person"] = focus_person
                mark_done_redirect = url_for("manual_video_detail", **redirect_args)  # type: ignore[arg-type]
            else:
                flash("Great job! No more videos need manual tagging right now.", "info")
                mark_done_redirect = url_for("manual_video_dashboard")

    if should_commit:
        conn.commit()
    if added:
        _invalidate_known_people_cache()
    if warmup_depth > 0 and next_manual_hashes:
        _schedule_manual_video_warmup_batch(next_manual_hashes)

    if mark_done_redirect:
        return redirect(mark_done_redirect)

    redirect_args = {"file_hash": file_hash}
    if focus_person:
        redirect_args["focus_person"] = focus_person
    return redirect(url_for("manual_video_detail", **redirect_args))


@app.route("/videos/manual/<file_hash>/tags/remove", methods=["POST"])
def manual_video_remove_tags(file_hash):
    """Removes selected tags from a manual video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)

    names = [name.strip() for name in request.form.getlist("person_name") if name.strip()]
    if not names:
        flash("Select at least one tag to remove.", "warning")
        redirect_args = {"file_hash": file_hash}
        focus_person = request.form.get("focus_person", "").strip()
        if focus_person:
            redirect_args["focus_person"] = focus_person
        return redirect(url_for("manual_video_detail", **redirect_args))

    conn.executemany(
        "DELETE FROM video_people WHERE file_hash = ? AND person_name = ?",
        [(file_hash, name) for name in names],
    )
    conn.commit()
    _invalidate_known_people_cache()
    flash(f"Removed tags: {', '.join(names)}", "success")
    redirect_args = {"file_hash": file_hash}
    focus_person = request.form.get("focus_person", "").strip()
    if focus_person:
        redirect_args["focus_person"] = focus_person
    return redirect(url_for("manual_video_detail", **redirect_args))


@app.route("/videos/manual/<file_hash>/status", methods=["POST"])
def manual_video_update_status(file_hash):
    """Updates the manual review status for a video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)

    status = request.form.get("status")
    if status not in MANUAL_FINAL_STATUSES:
        flash("Invalid status selection.", "error")
        return redirect(url_for("manual_video_detail", file_hash=file_hash))

    if status == "done":
        tag_count = conn.execute(
            "SELECT COUNT(*) as count FROM video_people WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()["count"]
        if tag_count == 0:
            flash(
                'Add at least one person before marking as done, or choose "No people".', "warning"
            )
            return redirect(url_for("manual_video_detail", file_hash=file_hash))

    conn.execute(
        "UPDATE scanned_files SET manual_review_status = ? WHERE file_hash = ?",
        (status, file_hash),
    )

    cleared_tags = False
    if status == "no_people":
        conn.execute("DELETE FROM video_people WHERE file_hash = ?", (file_hash,))
        cleared_tags = True

    conn.commit()
    if cleared_tags:
        _invalidate_known_people_cache()

    if status == "done":
        flash("Marked video as manually tagged.", "success")
    else:
        flash("Marked video as reviewed with no identifiable people.", "success")

    focus_person = request.form.get("focus_person", "").strip()

    warmup_depth = max(int(config.MANUAL_REVIEW_WARMUP_DEPTH or 0), 0)
    next_hashes = _get_next_manual_video_hashes(
        conn, exclude_hash=file_hash, limit=max(warmup_depth, 1)
    )
    if next_hashes:
        if warmup_depth > 0:
            _schedule_manual_video_warmup_batch(next_hashes)
        redirect_args = {"file_hash": next_hashes[0]}
        if focus_person:
            redirect_args["focus_person"] = focus_person
        return redirect(url_for("manual_video_detail", **redirect_args))  # type: ignore[arg-type]

    flash("Great job! No more videos need manual tagging right now.", "info")
    return redirect(url_for("manual_video_dashboard"))


@app.route("/videos/manual/<file_hash>/reshuffle", methods=["POST"])
def manual_video_reshuffle(file_hash):
    """Creates a new random set of sample frames for the video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)
    focus_person = request.form.get("focus_person", "").strip()
    samples, video_path = _get_video_samples(conn, file_hash, regenerate=True)
    if not samples:
        if not (video_path and os.path.exists(video_path)):
            flash("Original video is missing; cannot generate frames.", "error")
        else:
            flash("Could not generate sample frames. Try again later.", "error")
    else:
        flash("Generated a fresh set of frames.", "success")
    redirect_args = {"file_hash": file_hash}
    if focus_person:
        redirect_args["focus_person"] = focus_person
    return redirect(url_for("manual_video_detail", **redirect_args))


@app.route("/videos/manual/<file_hash>/frames/<path:filename>")
def manual_video_frame(file_hash, filename):
    """Serves generated frame samples for manual tagging."""
    _manual_feature_guard()
    sample_dir = _resolve_sample_dir(file_hash)
    base_dir = sample_dir.resolve()
    image_path = (sample_dir / filename).resolve()

    try:
        image_path.relative_to(base_dir)
    except ValueError:
        abort(404)

    if not image_path.exists():
        abort(404)

    return send_file(str(image_path), mimetype="image/jpeg")


@app.route("/name_cluster", methods=["POST"])
def name_cluster():
    """Handles the form submission for naming a cluster."""
    cluster_id = request.form["cluster_id"]
    person_name = request.form["person_name"].strip()

    if cluster_id and person_name:
        if person_name.lower() == "unknown":
            flash("'Unknown' is reserved. Use the button to mark as unknown instead.", "error")
            return redirect(url_for("tag_group", cluster_id=cluster_id))
        conn = get_db_connection()
        existing = conn.execute(
            "SELECT 1 FROM faces WHERE person_name = ? LIMIT 1", (person_name,)
        ).fetchone()
        if existing:
            flash(
                f"A person named '{person_name}' already exists. Use the merge option instead.",
                "error",
            )
            return redirect(url_for("tag_group", cluster_id=cluster_id))
        conn.execute(
            "UPDATE faces SET person_name = ? WHERE cluster_id = ?", (person_name, cluster_id)
        )
        conn.execute(
            """
            UPDATE faces
            SET suggestion_status = CASE
                    WHEN suggested_person_name = ? THEN 'accepted'
                    ELSE 'cleared'
                END,
                suggested_person_name = NULL,
                suggested_confidence = NULL,
                suggested_candidates = NULL
            WHERE cluster_id = ?
        """,
            (person_name, cluster_id),
        )
        conn.commit()
        flash(f"Assigned name '{person_name}' to cluster #{cluster_id}", "success")
    return redirect(url_for("index"))


@app.route("/mark_unknown", methods=["POST"])
def mark_unknown():
    """Marks the current cluster as Unknown, moving faces into the Unknown group."""
    cluster_id = request.form["cluster_id"]
    if not cluster_id:
        flash("Invalid cluster.", "error")
        return redirect(url_for("index"))

    cluster_id = int(cluster_id)
    if cluster_id == -1:
        flash("Cluster is already Unknown.", "info")
        return redirect(url_for("tag_group", cluster_id=cluster_id))

    conn = get_db_connection()

    # Assign person_name = 'Unknown'
    conn.execute("UPDATE faces SET person_name = ? WHERE cluster_id = ?", ("Unknown", cluster_id))

    # Move the faces to the Unknown cluster (-1)
    conn.execute("UPDATE faces SET cluster_id = -1 WHERE cluster_id = ?", (cluster_id,))

    conn.execute("""
        UPDATE faces
        SET suggestion_status = NULL,
            suggested_person_name = NULL,
            suggested_confidence = NULL,
            suggested_candidates = NULL
        WHERE cluster_id = -1
    """)

    conn.commit()

    flash(f"Marked cluster #{cluster_id} as Unknown.", "success")
    return redirect(url_for("index"))


def _delete_faces(conn, cluster_id, face_ids):
    placeholders = ", ".join("?" for _ in face_ids)
    conn.execute(f"DELETE FROM faces WHERE id IN ({placeholders})", face_ids)
    conn.commit()

    for fid in face_ids:
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{fid}.jpg")
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

    remaining = conn.execute(
        "SELECT 1 FROM faces WHERE cluster_id = ? LIMIT 1", (cluster_id,)
    ).fetchone()
    return remaining is not None


@app.route("/delete_cluster", methods=["POST"])
def delete_cluster():
    """Deletes all data associated with a cluster_id."""
    cluster_id = request.form["cluster_id"]
    if cluster_id:
        if int(cluster_id) == -1:
            flash("Cannot delete the Unknown group.", "error")
            return redirect(url_for("tag_group", cluster_id=cluster_id))
        conn = get_db_connection()
        conn.execute("DELETE FROM faces WHERE cluster_id = ?", (cluster_id,))
        conn.commit()
        flash(f"Deleted all faces for cluster #{cluster_id}", "success")
    return redirect(url_for("index"))


@app.route("/accept_suggestion", methods=["POST"])
def accept_suggestion():
    """Accepts an automatic naming suggestion for a cluster."""
    cluster_id = request.form.get("cluster_id")
    suggestion_name = request.form.get("suggestion_name", "").strip()

    if not cluster_id or not suggestion_name:
        flash("Invalid suggestion payload.", "error")
        return redirect(url_for("index"))

    conn = get_db_connection()
    conn.execute(
        "UPDATE faces SET person_name = ? WHERE cluster_id = ?", (suggestion_name, cluster_id)
    )
    conn.execute(
        """
        UPDATE faces
        SET suggestion_status = CASE
                WHEN suggested_person_name = ? THEN 'accepted'
                ELSE 'cleared'
            END,
            suggested_person_name = NULL,
            suggested_confidence = NULL,
            suggested_candidates = NULL
        WHERE cluster_id = ?
    """,
        (suggestion_name, cluster_id),
    )
    conn.commit()
    flash(f"Accepted suggestion '{suggestion_name}' for cluster #{cluster_id}.", "success")
    return redirect(url_for("index"))


@app.route("/reject_suggestion", methods=["POST"])
def reject_suggestion():
    """Rejects an automatic naming suggestion for a cluster."""
    cluster_id = request.form.get("cluster_id")
    suggestion_name = request.form.get("suggestion_name", "").strip()

    if not cluster_id or not suggestion_name:
        flash("Invalid suggestion payload.", "error")
        return redirect(url_for("index"))

    conn = get_db_connection()
    conn.execute(
        """
        UPDATE faces
        SET suggestion_status = 'rejected'
        WHERE cluster_id = ? AND suggested_person_name = ?
    """,
        (cluster_id, suggestion_name),
    )
    conn.commit()
    flash(f"Rejected suggestion '{suggestion_name}' for cluster #{cluster_id}.", "info")
    return redirect(url_for("tag_group", cluster_id=cluster_id))


@app.route("/skip_cluster/<int:cluster_id>")
def skip_cluster(cluster_id):
    """Marks a group as skipped so it is revisited after other groups."""
    skipped = session.get("skipped_clusters", [])
    if cluster_id not in skipped:
        skipped.append(cluster_id)
    session["skipped_clusters"] = skipped
    return redirect(url_for("index"))


@app.route("/metadata_preview")
def metadata_preview():
    """Displays the enhanced metadata planner interface."""
    return render_template("metadata_preview.html")


@app.route("/write_metadata", methods=["POST"])
def write_metadata():
    """Starts an asynchronous NFO metadata write operation and redirects to progress view."""
    selected_hashes = request.form.getlist("file_hashes")
    if not selected_hashes:
        flash("Select at least one video before starting the write operation.", "warning")
        return redirect(url_for("metadata_preview"))

    items = nfo_planner.build_plan(selected_hashes)

    selected_set = set(selected_hashes)
    ready_items: list[NfoPlanItem] = []
    blocked_items: list[NfoPlanItem] = []
    for item in items:
        if item.file_hash not in selected_set:
            continue
        if not item.requires_update:
            continue
        if not item.can_update:
            blocked_items.append(item)
            continue
        ready_items.append(item)

    if not ready_items:
        if blocked_items:
            flash(
                f"No metadata updates were started. {len(blocked_items)} file(s) have no NFO file.",
                "warning",
            )
        else:
            flash("No metadata updates were required for the selected files.", "info")
        return redirect(url_for("metadata_preview"))

    if blocked_items:
        flash(
            f"Skipping {len(blocked_items)} file(s) that have no NFO file. They will remain in the planner.",
            "warning",
        )

    try:
        operation_id = nfo_writer.start_operation(ready_items, backup=True)
    except Exception as exc:  # noqa: BLE001
        flash(f"Unable to start metadata write operation: {exc}", "error")
        return redirect(url_for("metadata_preview"))

    flash(
        f"Writing metadata for {len(ready_items)} file(s). You can monitor progress below.",
        "success",
    )
    return redirect(url_for("metadata_progress", operation_id=operation_id))


# --- NEW ROUTES FOR REVIEWING/EDITING ---


@app.route("/people")
def list_people():
    """Shows a grid of all identified people."""
    conn = get_db_connection()
    # Get one representative face for each person, plus a count of their faces
    people = conn.execute("""
        SELECT p.person_name, p.face_count, f.id as face_id
        FROM (
            SELECT person_name, COUNT(id) as face_count, MIN(id) as min_face_id
            FROM faces
            WHERE person_name IS NOT NULL
            GROUP BY person_name
        ) p
        JOIN faces f ON p.min_face_id = f.id
        ORDER BY p.person_name
    """).fetchall()
    return render_template("people_list.html", people=people)


@app.route("/person/<person_name>")
def person_details(person_name):
    """Shows all faces for one person and provides editing tools."""
    conn = get_db_connection()
    faces = conn.execute("SELECT id FROM faces WHERE person_name = ?", (person_name,)).fetchall()
    if not faces:
        flash(f"Person '{person_name}' not found.", "error")
        return redirect(url_for("list_people"))
    file_rows = conn.execute(
        """
        SELECT DISTINCT sf.file_hash, sf.last_known_filepath
        FROM faces f
        LEFT JOIN scanned_files sf ON f.file_hash = sf.file_hash
        WHERE f.person_name = ?
        ORDER BY LOWER(COALESCE(sf.last_known_filepath, ''))
    """,
        (person_name,),
    ).fetchall()
    files = [
        {
            "file_hash": row["file_hash"],
            "path": row["last_known_filepath"],
            "name": os.path.basename(row["last_known_filepath"])
            if row["last_known_filepath"]
            else None,
        }
        for row in file_rows
    ]
    return render_template("person_detail.html", person_name=person_name, faces=faces, files=files)


@app.route("/rename_person/<old_name>", methods=["POST"])
def rename_person(old_name):
    new_name = request.form["new_name"].strip()
    if not new_name:
        flash("New name cannot be empty.", "error")
        return redirect(url_for("person_details", person_name=old_name))
    if new_name.lower() == "unknown":
        flash("'Unknown' is reserved. Choose a different name.", "error")
        return redirect(url_for("person_details", person_name=old_name))
    conn = get_db_connection()
    existing = conn.execute(
        "SELECT 1 FROM faces WHERE person_name = ? LIMIT 1", (new_name,)
    ).fetchone()
    if existing:
        flash(f"A person named '{new_name}' already exists. Use merge instead.", "error")
        return redirect(url_for("person_details", person_name=old_name))
    conn.execute("UPDATE faces SET person_name = ? WHERE person_name = ?", (new_name, old_name))
    conn.commit()
    flash(f"Renamed '{old_name}' to '{new_name}'.", "success")
    return redirect(url_for("person_details", person_name=new_name))


@app.route("/unname_person", methods=["POST"])
def unname_person():
    person_name = request.form["person_name"]
    conn = get_db_connection()
    conn.execute("UPDATE faces SET person_name = NULL WHERE person_name = ?", (person_name,))
    conn.commit()
    flash(f"'{person_name}' has been un-named and their group is back in the queue.", "success")
    return redirect(url_for("list_people"))


@app.route("/delete_cluster_by_name", methods=["POST"])
def delete_cluster_by_name():
    person_name = request.form["person_name"]
    conn = get_db_connection()
    # This is safer than deleting by cluster_id, as a person might be a result of merges
    conn.execute("DELETE FROM faces WHERE person_name = ?", (person_name,))
    conn.commit()
    flash(f"Deleted person '{person_name}' and all their face data.", "success")
    return redirect(url_for("list_people"))


@app.route("/remove_person_faces", methods=["POST"])
def remove_person_faces():
    """Deletes selected faces for a person and their thumbnails."""
    person_name = request.form["person_name"]
    face_ids = request.form.getlist("face_ids")

    if not face_ids:
        flash("You must select at least one face to remove.", "error")
        return redirect(url_for("person_details", person_name=person_name))

    conn = get_db_connection()
    placeholders = ", ".join("?" for _ in face_ids)
    conn.execute(f"DELETE FROM faces WHERE id IN ({placeholders})", face_ids)
    conn.commit()

    for fid in face_ids:
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{fid}.jpg")
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

    remaining = conn.execute(
        "SELECT 1 FROM faces WHERE person_name = ? LIMIT 1", (person_name,)
    ).fetchone()

    flash(f"Removed {len(face_ids)} face(s).", "success")
    if remaining:
        return redirect(url_for("person_details", person_name=person_name))
    else:
        flash(
            f"'{person_name}' no longer has any faces and has been removed from the list.", "info"
        )
        return redirect(url_for("list_people"))


# --- NEW ROUTES FOR MERGE/SPLIT ---


@app.route("/merge_clusters", methods=["POST"])
def merge_clusters():
    from_cluster_id = request.form["from_cluster_id"]
    to_person_name = request.form.get("to_person_name")

    if not to_person_name:
        flash("You must select a person to merge with.", "error")
        return redirect(url_for("tag_group", cluster_id=from_cluster_id))

    if to_person_name.lower() == "unknown":
        flash("Cannot merge into the Unknown person.", "error")
        return redirect(url_for("tag_group", cluster_id=from_cluster_id))

    conn = get_db_connection()
    # Find the cluster_id of the person we are merging into
    target_cluster = conn.execute(
        "SELECT cluster_id FROM faces WHERE person_name = ? LIMIT 1", (to_person_name,)
    ).fetchone()
    if not target_cluster:
        flash(f"Could not find target person '{to_person_name}'.", "error")
        return redirect(url_for("tag_group", cluster_id=from_cluster_id))

    to_cluster_id = target_cluster["cluster_id"]

    # Reassign the old cluster_id and set the name
    conn.execute(
        "UPDATE faces SET cluster_id = ?, person_name = ? WHERE cluster_id = ?",
        (to_cluster_id, to_person_name, from_cluster_id),
    )
    conn.execute(
        """
        UPDATE faces
        SET suggestion_status = CASE
                WHEN suggested_person_name = ? THEN 'accepted'
                ELSE 'cleared'
            END,
            suggested_person_name = NULL,
            suggested_confidence = NULL,
            suggested_candidates = NULL
        WHERE cluster_id = ?
    """,
        (to_person_name, to_cluster_id),
    )
    conn.commit()

    flash(f"Successfully merged group #{from_cluster_id} into '{to_person_name}'.", "success")
    return redirect(url_for("index"))


@app.route("/split_cluster", methods=["POST"])
def split_cluster():
    original_cluster_id = request.form["cluster_id"]
    face_ids_to_split = request.form.getlist("face_ids")

    if not face_ids_to_split:
        flash("You must select at least one face to split into a new group.", "error")
        return redirect(url_for("tag_group", cluster_id=original_cluster_id))

    conn = get_db_connection()
    # Find the highest existing cluster_id to create a new one
    max_cluster_id_result = conn.execute("SELECT MAX(cluster_id) FROM faces").fetchone()
    new_cluster_id = (max_cluster_id_result[0] or 0) + 1

    # Create a placeholder string for the SQL query
    placeholders = ", ".join("?" for _ in face_ids_to_split)
    query = f"UPDATE faces SET cluster_id = ? WHERE id IN ({placeholders})"

    params = [new_cluster_id] + face_ids_to_split
    conn.execute(query, params)
    conn.commit()

    flash(
        f"Successfully split {len(face_ids_to_split)} faces from group #{original_cluster_id} into new group #{new_cluster_id}.",
        "success",
    )
    return redirect(url_for("tag_group", cluster_id=original_cluster_id))


@app.route("/remove_faces", methods=["POST"])
def remove_faces():
    """Deletes selected faces and their thumbnails."""
    cluster_id = request.form["cluster_id"]
    face_ids = request.form.getlist("face_ids")

    if not face_ids:
        flash("You must select at least one face to remove.", "error")
        return redirect(url_for("tag_group", cluster_id=cluster_id))

    conn = get_db_connection()
    remaining = _delete_faces(conn, cluster_id, face_ids)

    flash(f"Removed {len(face_ids)} face(s).", "success")
    if remaining:
        return redirect(url_for("tag_group", cluster_id=cluster_id))
    else:
        flash(f"Cluster #{cluster_id} is now empty and has been removed.", "info")
        return redirect(url_for("index"))


@app.route("/delete_selected_faces", methods=["POST"])
def delete_selected_faces():
    """Deletes selected faces entirely from the database and disk."""
    cluster_id = request.form["cluster_id"]
    face_ids = request.form.getlist("face_ids")

    if not face_ids:
        flash("You must select at least one face to delete.", "error")
        return redirect(url_for("tag_group", cluster_id=cluster_id))

    conn = get_db_connection()
    remaining = _delete_faces(conn, cluster_id, face_ids)

    flash(f"Deleted {len(face_ids)} face(s).", "success")
    if remaining:
        return redirect(url_for("tag_group", cluster_id=cluster_id))
    else:
        flash(f"Cluster #{cluster_id} is now empty and has been removed.", "info")
        return redirect(url_for("index"))


@app.route("/remove_video_faces/<int:cluster_id>/<file_hash>", methods=["POST"])
def remove_video_faces(cluster_id, file_hash):
    """Removes all faces from a specific video in a cluster and creates a new cluster for them."""
    conn = get_db_connection()

    # Get all faces in the cluster that belong to the specified video
    faces_to_move = conn.execute(
        "SELECT id FROM faces WHERE cluster_id = ? AND file_hash = ?", (cluster_id, file_hash)
    ).fetchall()

    if not faces_to_move:
        flash("No faces found from this video in the current group.", "warning")
        return redirect(url_for("tag_group", cluster_id=cluster_id))

    # Generate a new cluster_id
    max_cluster_id_result = conn.execute("SELECT MAX(cluster_id) FROM faces").fetchone()
    new_cluster_id = (max_cluster_id_result[0] or 0) + 1

    # Move the faces to the new cluster
    face_ids = [face["id"] for face in faces_to_move]
    placeholders = ", ".join("?" for _ in face_ids)
    conn.execute(
        f"UPDATE faces SET cluster_id = ? WHERE id IN ({placeholders})", [new_cluster_id] + face_ids
    )
    conn.commit()

    flash(
        f"Removed {len(face_ids)} face(s) from this video and created new group #{new_cluster_id}.",
        "success",
    )
    return redirect(url_for("tag_group", cluster_id=cluster_id))


if __name__ == "__main__":
    # Make sure the web app is accessible from other devices on your network
    app.run(host="0.0.0.0", port=5001, debug=config.DEBUG)
