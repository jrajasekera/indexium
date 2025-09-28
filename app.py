import json
import math
import os
import random
import re
import secrets
import sqlite3
from pathlib import Path

import ffmpeg
import cv2
from flask import (
    Flask,
    abort,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    g,
    session,
)

from config import Config

config = Config()

app = Flask(__name__)
# Flask needs a secret key to use flash messages
app.secret_key = config.SECRET_KEY

# --- DATABASE & HELPERS ---

def get_db_connection():
    """Gets a per-request database connection."""
    if "db" not in g:
        g.db = sqlite3.connect(config.DATABASE_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db_connection(exception):
    """Closes the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


MANUAL_ACTIVE_STATUSES = {"pending", "in_progress"}
MANUAL_FINAL_STATUSES = {"done", "no_people"}
MANUAL_ALL_STATUSES = MANUAL_ACTIVE_STATUSES | MANUAL_FINAL_STATUSES | {"not_required"}
SAMPLE_MAX_WIDTH = 640
SAMPLE_MAX_HEIGHT = 360
MIN_TEXT_FRAGMENT_LENGTH = 4


def _calculate_top_text_fragments(entries, max_results=5):
    """Return the top substrings by frequency and length from OCR text entries."""
    if not entries:
        return []

    substring_data: dict[str, dict[str, object]] = {}
    min_length = MIN_TEXT_FRAGMENT_LENGTH

    for raw_text, weight in entries:
        if not raw_text:
            continue
        weight = max(int(weight or 0), 0)
        if weight == 0:
            weight = 1

        normalized = raw_text.lower()
        if len(normalized) < min_length:
            continue

        per_string_seen: set[str] = set()
        for start in range(len(normalized) - min_length + 1):
            for end in range(start + min_length, len(normalized) + 1):
                substring_lower = normalized[start:end]
                if substring_lower in per_string_seen:
                    continue
                per_string_seen.add(substring_lower)

                display_slice = raw_text[start:end]
                info = substring_data.get(substring_lower)
                if info is None:
                    info = {
                        "count": 0,
                        "length": end - start,
                        "display": display_slice,
                    }
                    substring_data[substring_lower] = info

                info["count"] = int(info["count"]) + weight

    if not substring_data:
        return []

    ranked = sorted(
        (
            {
                "substring": info["display"],
                "lower": key,
                "count": int(info["count"]),
                "length": int(info["length"]),
            }
            for key, info in substring_data.items()
        ),
        key=lambda item: (-item["count"], -item["length"], item["lower"]),
    )

    return ranked[:max_results]


def _manual_feature_guard():
    """Abort with 404 if manual video review is disabled."""
    if not config.MANUAL_VIDEO_REVIEW_ENABLED:
        abort(404)


def _manual_status_sort_key(status):
    order = {
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


def _ensure_sample_seed(conn, file_hash: str, regenerate: bool = False) -> int | None:
    """Get or create the deterministic sampling seed for a video."""
    row = conn.execute(
        "SELECT sample_seed FROM scanned_files WHERE file_hash = ?",
        (file_hash,),
    ).fetchone()
    if not row:
        return None
    seed = row["sample_seed"]
    if regenerate or seed is None:
        seed = secrets.randbits(32)
        conn.execute(
            "UPDATE scanned_files SET sample_seed = ? WHERE file_hash = ?",
            (seed, file_hash),
        )
        conn.commit()
    return seed


def _resize_frame_for_sample(frame):
    """Resize frames for display while keeping aspect ratio."""
    height, width = frame.shape[:2]
    if not height or not width:
        return frame
    scale = min(SAMPLE_MAX_WIDTH / width, SAMPLE_MAX_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _generate_video_samples(video_path: str, sample_dir: Path, sample_count: int, seed: int) -> list[Path]:
    """Generate sample frames from a video and persist them to disk."""
    generated: list[Path] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        app.logger.warning("Could not open video %s for manual sampling", video_path)
        return generated

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            app.logger.warning("Video %s reports no frame count; falling back to sequential sampling", video_path)
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


def _get_video_samples(conn, file_hash: str, regenerate: bool = False) -> tuple[list[Path], str | None]:
    """Ensure sample frames exist for a video and return them with the source path."""
    row = conn.execute(
        "SELECT last_known_filepath FROM scanned_files WHERE file_hash = ?",
        (file_hash,),
    ).fetchone()
    if not row:
        return [], None

    video_path = row["last_known_filepath"]
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

    if existing_paths and not regenerate and len(existing_paths) >= min(sample_count, len(existing_paths)):
        return existing_paths[:sample_count], video_path

    seed = _ensure_sample_seed(conn, file_hash, regenerate or not existing_paths)
    if seed is None:
        return existing_paths[:sample_count], video_path

    generated = _generate_video_samples(video_path, sample_dir, sample_count, seed)
    if not generated and existing_paths:
        return existing_paths[:sample_count], video_path
    return generated, video_path


def _collect_known_people(conn) -> list[str]:
    """Return a sorted list of known people names from faces and manual video tags."""
    names = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL"
        ).fetchall()
    }
    names.update(
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT person_name FROM video_people"
        ).fetchall()
    )
    return sorted(names, key=lambda name: name.lower())


def _get_manual_video_record(conn, file_hash: str):
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


def _get_next_manual_video(conn, exclude_hash: str | None = None) -> str | None:
    """Return the next manual-review video hash, optionally skipping current."""
    query = [
        """
        SELECT file_hash
        FROM scanned_files
        WHERE manual_review_status IN ('pending', 'in_progress')
        """
    ]
    params: list[str] = []
    if exclude_hash:
        query.append("AND file_hash != ?")
        params.append(exclude_hash)
    query.append(
        "ORDER BY CASE manual_review_status WHEN 'pending' THEN 0 ELSE 1 END,"
        "         COALESCE(last_attempt, CURRENT_TIMESTAMP) ASC,"
        "         file_hash ASC"
    )
    query.append("LIMIT 1")
    row = conn.execute("\n".join(query), params).fetchone()
    return row['file_hash'] if row else None

def extract_people_from_comment(comment):
    """Extracts a set of person names from a metadata comment field."""
    if not comment:
        return set()

    match = re.search(r"People:\s*(.*)", comment)
    if not match:
        return set()

    people_segment = match.group(1)
    for terminator in ('\n', ';', '|'):
        if terminator in people_segment:
            people_segment = people_segment.split(terminator, 1)[0]
    names = [name.strip() for name in people_segment.split(',') if name.strip()]
    return set(names)


def build_metadata_plan(conn, target_hashes=None):
    """Builds a per-file plan summarizing pending metadata writes."""
    params = []
    query = 'SELECT DISTINCT file_hash FROM faces WHERE person_name IS NOT NULL'
    if target_hashes:
        placeholders = ','.join('?' for _ in target_hashes)
        query += f' AND file_hash IN ({placeholders})'
        params.extend(target_hashes)

    videos = conn.execute(query, params).fetchall()

    plan = []
    for video in videos:
        file_hash = video['file_hash']
        db_names_rows = conn.execute(
            'SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL',
            (file_hash,)
        ).fetchall()
        db_people = sorted({row['person_name'] for row in db_names_rows})

        path_row = conn.execute(
            'SELECT last_known_filepath FROM scanned_files WHERE file_hash = ?',
            (file_hash,)
        ).fetchone()
        raw_path = path_row['last_known_filepath'] if path_row else None
        can_update = bool(raw_path and os.path.exists(raw_path))

        existing_comment = None
        existing_people = set()
        probe_error = None

        if can_update:
            try:
                probe = ffmpeg.probe(raw_path)
                existing_comment = probe.get('format', {}).get('tags', {}).get('comment')
            except ffmpeg.Error as exc:
                probe_error = exc.stderr.decode('utf8') if getattr(exc, 'stderr', None) else str(exc)
                existing_comment = ''

        existing_comment_value = (existing_comment or '').strip()
        if existing_comment_value:
            existing_people = extract_people_from_comment(existing_comment_value)

        result_people = sorted(set(db_people).union(existing_people))
        if not result_people:
            continue

        result_comment = f"People: {', '.join(result_people)}"
        tags_to_add = sorted(set(db_people) - existing_people)
        metadata_only_people = sorted(existing_people - set(db_people))

        requires_update = existing_comment_value != result_comment or not can_update
        will_overwrite_comment = bool(existing_comment_value and existing_comment_value != result_comment)
        overwrites_custom_comment = bool(existing_comment_value and not existing_comment_value.startswith('People:'))

        plan.append({
            'file_hash': file_hash,
            'path': raw_path,
            'name': os.path.basename(raw_path) if raw_path else None,
            'db_people': db_people,
            'existing_people': sorted(existing_people),
            'metadata_only_people': metadata_only_people,
            'result_people': result_people,
            'tags_to_add': tags_to_add,
            'existing_comment': existing_comment if existing_comment is not None else None,
            'result_comment': result_comment,
            'requires_update': requires_update,
            'can_update': can_update,
            'will_overwrite_comment': will_overwrite_comment if existing_comment is not None else None,
            'overwrites_custom_comment': overwrites_custom_comment if existing_comment is not None else None,
            'probe_error': probe_error,
        })

    plan.sort(key=lambda item: ((item['name'] or '').lower(), item['file_hash']))
    return plan


def get_progress_stats():
    """Gets counts for UI display."""
    conn = get_db_connection()
    unnamed_groups_count = conn.execute('''
        SELECT COUNT(DISTINCT cluster_id) as count 
        FROM faces 
        WHERE cluster_id IS NOT NULL AND person_name IS NULL
    ''').fetchone()['count']
    named_people_count = conn.execute('''
        SELECT COUNT(DISTINCT person_name) as count 
        FROM faces 
        WHERE person_name IS NOT NULL
    ''').fetchone()['count']
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

@app.route('/')
def index():
    """Finds the next unnamed group and redirects to the tagging page for it."""
    conn = get_db_connection()
    skipped = session.get('skipped_clusters', [])

    base_query = '''
        SELECT MIN(cluster_id) as id
        FROM faces
        WHERE cluster_id IS NOT NULL
          AND person_name IS NULL
          AND cluster_id != ?
    '''

    params = []
    params.append(-1)
    if skipped:
        placeholders = ','.join('?' for _ in skipped)
        query = base_query + f" AND cluster_id NOT IN ({placeholders})"
        params.extend(skipped)
    else:
        query = base_query

    next_group = conn.execute(query, params).fetchone()

    if (not next_group or next_group['id'] is None) and skipped:
        session.pop('skipped_clusters', None)
        next_group = conn.execute(base_query, ( -1, )).fetchone()

    if next_group and next_group['id'] is not None:
        return redirect(url_for('tag_group', cluster_id=next_group['id']))
    else:
        # No more groups to name, show a completion page
        return render_template('all_done.html')


@app.route('/group/<int:cluster_id>')
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

    file_rows = conn.execute('''
        SELECT DISTINCT sf.last_known_filepath, sf.file_hash
        FROM faces f
        JOIN scanned_files sf ON f.file_hash = sf.file_hash
        WHERE f.cluster_id = ?
    ''', (cluster_id,)).fetchall()
    file_names = [os.path.basename(row['last_known_filepath']) for row in file_rows]
    file_hashes = [row['file_hash'] for row in file_rows]
    files_data = list(zip(file_names, file_hashes))
    file_name_by_hash = {file_hash: name for name, file_hash in files_data}

    ocr_aggregated = []
    ocr_by_file = {}
    ocr_top_fragments = []
    if file_hashes:
        placeholders = ','.join('?' for _ in file_hashes)
        rows = conn.execute(
            f'''
                SELECT vt.file_hash, vt.raw_text, vt.occurrence_count
                FROM video_text vt
                WHERE vt.file_hash IN ({placeholders})
            ''',
            file_hashes,
        ).fetchall()

        dedup_global = {}
        weighted_entries = []
        for row in rows:
            raw_text = (row['raw_text'] or '').strip()
            if not raw_text or len(raw_text) < MIN_TEXT_FRAGMENT_LENGTH:
                continue
            normalized = raw_text.lower()
            weight = row['occurrence_count'] or 1

            per_file = ocr_by_file.setdefault(row['file_hash'], [])
            if all(text.lower() != normalized for text in per_file):
                per_file.append(raw_text)

            entry = dedup_global.setdefault(normalized, {'raw_text': raw_text, 'file_hashes': set()})
            entry['raw_text'] = raw_text
            entry['file_hashes'].add(row['file_hash'])
            weighted_entries.append((raw_text, weight))

        for file_hash, values in ocr_by_file.items():
            values.sort(key=lambda text: text.lower())

        ocr_aggregated = []
        for data in dedup_global.values():
            file_hash_list = sorted(data['file_hashes'])
            ocr_aggregated.append(
                {
                    'raw_text': data['raw_text'],
                    'file_hashes': file_hash_list,
                    'file_count': len(file_hash_list),
                    'file_names': [file_name_by_hash.get(fh, fh) for fh in file_hash_list],
                }
            )
        ocr_aggregated.sort(key=lambda item: item['raw_text'].lower())

        ocr_top_fragments = _calculate_top_text_fragments(weighted_entries)

    ocr_text_by_file_items = [
        {
            'file_hash': file_hash,
            'file_name': file_name_by_hash.get(file_hash, file_hash),
            'texts': texts,
        }
        for file_hash, texts in ocr_by_file.items()
    ]
    ocr_text_by_file_items.sort(key=lambda item: item['file_name'].lower())

    if not sample_faces:
        flash(f"Cluster #{cluster_id} no longer exists or is empty.", "error")
        return redirect(url_for('index'))

    names = conn.execute(
        'SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL ORDER BY person_name').fetchall()

    existing_names = [name['person_name'] for name in names]

    suggestion_rows = conn.execute(
        "SELECT suggested_candidates FROM faces WHERE cluster_id = ? AND suggested_candidates IS NOT NULL",
        (cluster_id,)
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
            name = candidate.get('name')
            confidence = candidate.get('confidence')
            if not name or confidence is None:
                continue
            bucket = candidate_summary.setdefault(name, {
                'name': name,
                'total_confidence': 0.0,
                'count': 0,
                'max_confidence': 0.0,
            })
            bucket['total_confidence'] += confidence
            bucket['count'] += 1
            bucket['max_confidence'] = max(bucket['max_confidence'], confidence)

    suggestion_candidates = []
    for bucket in candidate_summary.values():
        avg_confidence = bucket['total_confidence'] / bucket['count']
        suggestion_candidates.append({
            'name': bucket['name'],
            'avg_confidence': avg_confidence,
            'max_confidence': bucket['max_confidence'],
            'count': bucket['count'],
        })

    suggestion_candidates.sort(
        key=lambda item: (item['avg_confidence'], item['max_confidence'], item['count']),
        reverse=True,
    )
    suggestion_candidates = suggestion_candidates[:5]
    primary_suggestion = suggestion_candidates[0] if suggestion_candidates else None

    cluster_data = {
        'id': cluster_id,
        'faces': sample_faces,
        'page': page,
        'total_pages': total_pages,
    }
    return render_template('group_tagger.html',
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
                           cluster_is_unknown=(cluster_id == -1))


@app.route('/face_thumbnail/<int:face_id>')
def get_face_thumbnail(face_id):
    """Serves a pre-generated face thumbnail."""
    thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{face_id}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype='image/jpeg')
    return "Thumbnail not found", 404


# --- MANUAL VIDEO TAGGING ROUTES ---


@app.route('/videos/manual')
def manual_video_dashboard():
    """Lists videos that need manual people tagging."""
    _manual_feature_guard()
    conn = get_db_connection()
    focus_person = request.args.get('person', '').strip()
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

    tag_rows = conn.execute(
        "SELECT file_hash, person_name FROM video_people"
    ).fetchall()
    tags_by_file = {}
    for row in tag_rows:
        tags_by_file.setdefault(row['file_hash'], []).append(row['person_name'])

    videos = []
    for row in rows:
        video_path = row['last_known_filepath']
        tags = sorted(tags_by_file.get(row['file_hash'], []), key=lambda name: name.lower())
        matches_focus = bool(focus_person_normalized) and any(
            tag.lower() == focus_person_normalized for tag in tags
        )
        needs_focus = bool(focus_person_normalized) and not matches_focus and row['manual_review_status'] in MANUAL_ACTIVE_STATUSES
        videos.append({
            'file_hash': row['file_hash'],
            'name': os.path.basename(video_path) if video_path else None,
            'path': video_path,
            'status': row['manual_review_status'],
            'tag_count': row['tag_count'],
            'face_count': row['face_count'],
            'missing': not (video_path and os.path.exists(video_path)),
            'last_attempt': row['last_attempt'],
            'tags': tags,
            'matches_focus': matches_focus,
            'needs_focus': needs_focus,
        })

    counts = {status: 0 for status in MANUAL_ALL_STATUSES}
    for video in videos:
        counts[video['status']] = counts.get(video['status'], 0) + 1
    counts['total'] = len(videos)

    def status_order(status):
        return {
            'pending': 0,
            'in_progress': 1,
            'done': 2,
            'no_people': 3,
            'not_required': 4,
        }.get(status, 5)

    next_video = next((video for video in videos if video['status'] in MANUAL_ACTIVE_STATUSES), None)

    if focus_person:
        matching_videos = [video for video in videos if video['matches_focus']]
        suggested_videos = [video for video in videos if video['needs_focus']]
        other_videos = [
            video for video in videos
            if video not in matching_videos and video not in suggested_videos
        ]
    else:
        matching_videos = []
        suggested_videos = []
        other_videos = videos

    videos.sort(key=lambda video: (status_order(video['status']), (video['name'] or '').lower(), video['file_hash']))

    matching_videos.sort(key=lambda video: (status_order(video['status']), (video['name'] or '').lower(), video['file_hash']))
    suggested_videos.sort(key=lambda video: (status_order(video['status']), (video['name'] or '').lower(), video['file_hash']))
    other_videos.sort(key=lambda video: (status_order(video['status']), (video['name'] or '').lower(), video['file_hash']))

    known_people = _collect_known_people(conn)

    return render_template(
        'video_manual_list.html',
        videos=videos,
        counts=counts,
        next_video=next_video,
        focus_person=focus_person,
        matching_videos=matching_videos,
        suggested_videos=suggested_videos,
        other_videos=other_videos,
        known_people=known_people,
    )


@app.route('/videos/manual/next')
def manual_video_next():
    """Redirects to the next video requiring manual tagging."""
    _manual_feature_guard()
    conn = get_db_connection()
    next_hash = _get_next_manual_video(conn)
    if not next_hash:
        flash('No videos waiting for manual tagging.', 'info')
        return redirect(url_for('manual_video_dashboard'))
    return redirect(url_for('manual_video_detail', file_hash=next_hash))


@app.route('/videos/manual/<file_hash>')
def manual_video_detail(file_hash):
    """Displays sampling frames and tagging controls for a single video."""
    _manual_feature_guard()
    conn = get_db_connection()
    record = _get_manual_video_record(conn, file_hash)
    focus_person = request.args.get('focus_person', '').strip()

    status = record['manual_review_status']
    if status == 'not_required':
        flash('Video no longer requires manual tagging.', 'info')
        return redirect(url_for('manual_video_dashboard'))
    if status == 'pending':
        conn.execute(
            "UPDATE scanned_files SET manual_review_status = 'in_progress' WHERE file_hash = ?",
            (file_hash,),
        )
        conn.commit()
        status = 'in_progress'

    sample_paths, video_path = _get_video_samples(conn, file_hash)
    sample_files = [
        {
            'filename': path.name,
            'url': url_for('manual_video_frame', file_hash=file_hash, filename=path.name),
        }
        for path in sample_paths
    ]

    tags = [
        row['person_name']
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
    weighted_entries = []
    for row in text_rows:
        raw_text = (row['raw_text'] or '').strip()
        if not raw_text or len(raw_text) < MIN_TEXT_FRAGMENT_LENGTH:
            continue
        normalized = raw_text.lower()
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        occurrence = row['occurrence_count'] or 1
        weighted_entries.append((raw_text, occurrence))
        ocr_entries.append(
            {
                'raw_text': raw_text,
                'confidence': row['confidence'],
                'occurrence_count': occurrence,
            }
        )

    known_people = _collect_known_people(conn)
    video_info = {
        'file_hash': file_hash,
        'name': os.path.basename(video_path) if video_path else None,
        'path': video_path,
        'status': status,
        'face_count': record['face_count'],
        'tag_count': len(tags),
        'ocr_text_count': record['ocr_text_count'] or 0,
        'ocr_last_updated': record['ocr_last_updated'],
        'missing': not (video_path and os.path.exists(video_path)),
    }

    return render_template(
        'video_manual_detail.html',
        video=video_info,
        sample_images=sample_files,
        tags=tags,
        known_people=known_people,
        sample_count=len(sample_files),
        focus_person=focus_person,
        video_text_entries=ocr_entries,
        video_text_top_fragments=_calculate_top_text_fragments(weighted_entries),
    )


@app.route('/videos/manual/<file_hash>/tags', methods=['POST'])
def manual_video_add_tags(file_hash):
    """Adds a single tag to a manual video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)

    submitted_name = request.form.get('person_name', '').strip()

    if not submitted_name:
        flash('Provide a name to add.', 'warning')
        redirect_args = {'file_hash': file_hash}
        focus_person = request.form.get('focus_person', '').strip()
        if focus_person:
            redirect_args['focus_person'] = focus_person
        return redirect(url_for('manual_video_detail', **redirect_args))

    added = []
    duplicates = []
    invalid = []
    name = submitted_name
    if name.lower() == 'unknown':
        invalid.append(name)
    else:
        try:
            conn.execute(
                'INSERT INTO video_people (file_hash, person_name) VALUES (?, ?)',
                (file_hash, name),
            )
            added.append(name)
        except sqlite3.IntegrityError:
            duplicates.append(name)

    if added or invalid or duplicates:
        conn.commit()

    if added:
        flash(f"Tagged video with: {', '.join(added)}", 'success')
    if duplicates:
        flash(f"Already tagged: {', '.join(duplicates)}", 'info')
    if invalid:
        flash("'Unknown' is reserved. Use the review buttons instead.", 'error')

    redirect_args = {'file_hash': file_hash}
    focus_person = request.form.get('focus_person', '').strip()
    if focus_person:
        redirect_args['focus_person'] = focus_person
    return redirect(url_for('manual_video_detail', **redirect_args))


@app.route('/videos/manual/<file_hash>/tags/remove', methods=['POST'])
def manual_video_remove_tags(file_hash):
    """Removes selected tags from a manual video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)

    names = [name.strip() for name in request.form.getlist('person_name') if name.strip()]
    if not names:
        flash('Select at least one tag to remove.', 'warning')
        redirect_args = {'file_hash': file_hash}
        focus_person = request.form.get('focus_person', '').strip()
        if focus_person:
            redirect_args['focus_person'] = focus_person
        return redirect(url_for('manual_video_detail', **redirect_args))

    conn.executemany(
        'DELETE FROM video_people WHERE file_hash = ? AND person_name = ?',
        [(file_hash, name) for name in names],
    )
    conn.commit()
    flash(f"Removed tags: {', '.join(names)}", 'success')
    redirect_args = {'file_hash': file_hash}
    focus_person = request.form.get('focus_person', '').strip()
    if focus_person:
        redirect_args['focus_person'] = focus_person
    return redirect(url_for('manual_video_detail', **redirect_args))


@app.route('/videos/manual/<file_hash>/status', methods=['POST'])
def manual_video_update_status(file_hash):
    """Updates the manual review status for a video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)

    status = request.form.get('status')
    if status not in MANUAL_FINAL_STATUSES:
        flash('Invalid status selection.', 'error')
        return redirect(url_for('manual_video_detail', file_hash=file_hash))

    if status == 'done':
        tag_count = conn.execute(
            'SELECT COUNT(*) as count FROM video_people WHERE file_hash = ?',
            (file_hash,),
        ).fetchone()['count']
        if tag_count == 0:
            flash('Add at least one person before marking as done, or choose "No people".', 'warning')
            return redirect(url_for('manual_video_detail', file_hash=file_hash))

    conn.execute(
        'UPDATE scanned_files SET manual_review_status = ? WHERE file_hash = ?',
        (status, file_hash),
    )

    if status == 'no_people':
        conn.execute('DELETE FROM video_people WHERE file_hash = ?', (file_hash,))

    conn.commit()

    if status == 'done':
        flash('Marked video as manually tagged.', 'success')
    else:
        flash('Marked video as reviewed with no identifiable people.', 'success')

    focus_person = request.form.get('focus_person', '').strip()

    next_hash = _get_next_manual_video(conn, exclude_hash=file_hash)
    if next_hash:
        redirect_args = {'file_hash': next_hash}
        if focus_person:
            redirect_args['focus_person'] = focus_person
        return redirect(url_for('manual_video_detail', **redirect_args))

    flash('Great job! No more videos need manual tagging right now.', 'info')
    return redirect(url_for('manual_video_dashboard'))


@app.route('/videos/manual/<file_hash>/reshuffle', methods=['POST'])
def manual_video_reshuffle(file_hash):
    """Creates a new random set of sample frames for the video."""
    _manual_feature_guard()
    conn = get_db_connection()
    _get_manual_video_record(conn, file_hash)
    focus_person = request.form.get('focus_person', '').strip()
    samples, video_path = _get_video_samples(conn, file_hash, regenerate=True)
    if not samples:
        if not (video_path and os.path.exists(video_path)):
            flash('Original video is missing; cannot generate frames.', 'error')
        else:
            flash('Could not generate sample frames. Try again later.', 'error')
    else:
        flash('Generated a fresh set of frames.', 'success')
    redirect_args = {'file_hash': file_hash}
    if focus_person:
        redirect_args['focus_person'] = focus_person
    return redirect(url_for('manual_video_detail', **redirect_args))


@app.route('/videos/manual/<file_hash>/frames/<path:filename>')
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

    return send_file(str(image_path), mimetype='image/jpeg')
@app.route('/name_cluster', methods=['POST'])
def name_cluster():
    """Handles the form submission for naming a cluster."""
    cluster_id = request.form['cluster_id']
    person_name = request.form['person_name'].strip()

    if cluster_id and person_name:
        if person_name.lower() == 'unknown':
            flash("'Unknown' is reserved. Use the button to mark as unknown instead.", "error")
            return redirect(url_for('tag_group', cluster_id=cluster_id))
        conn = get_db_connection()
        conn.execute('UPDATE faces SET person_name = ? WHERE cluster_id = ?', (person_name, cluster_id))
        conn.execute('''
            UPDATE faces
            SET suggestion_status = CASE
                    WHEN suggested_person_name = ? THEN 'accepted'
                    ELSE 'cleared'
                END,
                suggested_person_name = NULL,
                suggested_confidence = NULL,
                suggested_candidates = NULL
            WHERE cluster_id = ?
        ''', (person_name, cluster_id))
        conn.commit()
        flash(f"Assigned name '{person_name}' to cluster #{cluster_id}", "success")
    return redirect(url_for('index'))


@app.route('/mark_unknown', methods=['POST'])
def mark_unknown():
    """Marks the current cluster as Unknown, moving faces into the Unknown group."""
    cluster_id = request.form['cluster_id']
    if not cluster_id:
        flash("Invalid cluster.", "error")
        return redirect(url_for('index'))

    cluster_id = int(cluster_id)
    if cluster_id == -1:
        flash("Cluster is already Unknown.", "info")
        return redirect(url_for('tag_group', cluster_id=cluster_id))

    conn = get_db_connection()

    # Assign person_name = 'Unknown'
    conn.execute('UPDATE faces SET person_name = ? WHERE cluster_id = ?', ('Unknown', cluster_id))

    # Move the faces to the Unknown cluster (-1)
    conn.execute('UPDATE faces SET cluster_id = -1 WHERE cluster_id = ?', (cluster_id,))

    conn.execute('''
        UPDATE faces
        SET suggestion_status = NULL,
            suggested_person_name = NULL,
            suggested_confidence = NULL,
            suggested_candidates = NULL
        WHERE cluster_id = -1
    ''')

    conn.commit()

    flash(f"Marked cluster #{cluster_id} as Unknown.", "success")
    return redirect(url_for('index'))


def _delete_faces(conn, cluster_id, face_ids):
    placeholders = ', '.join('?' for _ in face_ids)
    conn.execute(f'DELETE FROM faces WHERE id IN ({placeholders})', face_ids)
    conn.commit()

    for fid in face_ids:
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{fid}.jpg")
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

    remaining = conn.execute('SELECT 1 FROM faces WHERE cluster_id = ? LIMIT 1', (cluster_id,)).fetchone()
    return remaining is not None


@app.route('/delete_cluster', methods=['POST'])
def delete_cluster():
    """Deletes all data associated with a cluster_id."""
    cluster_id = request.form['cluster_id']
    if cluster_id:
        if int(cluster_id) == -1:
            flash("Cannot delete the Unknown group.", "error")
            return redirect(url_for('tag_group', cluster_id=cluster_id))
        conn = get_db_connection()
        conn.execute('DELETE FROM faces WHERE cluster_id = ?', (cluster_id,))
        conn.commit()
        flash(f"Deleted all faces for cluster #{cluster_id}", "success")
    return redirect(url_for('index'))


@app.route('/accept_suggestion', methods=['POST'])
def accept_suggestion():
    """Accepts an automatic naming suggestion for a cluster."""
    cluster_id = request.form.get('cluster_id')
    suggestion_name = request.form.get('suggestion_name', '').strip()

    if not cluster_id or not suggestion_name:
        flash("Invalid suggestion payload.", "error")
        return redirect(url_for('index'))

    conn = get_db_connection()
    conn.execute('UPDATE faces SET person_name = ? WHERE cluster_id = ?', (suggestion_name, cluster_id))
    conn.execute('''
        UPDATE faces
        SET suggestion_status = CASE
                WHEN suggested_person_name = ? THEN 'accepted'
                ELSE 'cleared'
            END,
            suggested_person_name = NULL,
            suggested_confidence = NULL,
            suggested_candidates = NULL
        WHERE cluster_id = ?
    ''', (suggestion_name, cluster_id))
    conn.commit()
    flash(f"Accepted suggestion '{suggestion_name}' for cluster #{cluster_id}.", "success")
    return redirect(url_for('index'))


@app.route('/reject_suggestion', methods=['POST'])
def reject_suggestion():
    """Rejects an automatic naming suggestion for a cluster."""
    cluster_id = request.form.get('cluster_id')
    suggestion_name = request.form.get('suggestion_name', '').strip()

    if not cluster_id or not suggestion_name:
        flash("Invalid suggestion payload.", "error")
        return redirect(url_for('index'))

    conn = get_db_connection()
    conn.execute('''
        UPDATE faces
        SET suggestion_status = 'rejected'
        WHERE cluster_id = ? AND suggested_person_name = ?
    ''', (cluster_id, suggestion_name))
    conn.commit()
    flash(f"Rejected suggestion '{suggestion_name}' for cluster #{cluster_id}.", "info")
    return redirect(url_for('tag_group', cluster_id=cluster_id))


@app.route('/skip_cluster/<int:cluster_id>')
def skip_cluster(cluster_id):
    """Marks a group as skipped so it is revisited after other groups."""
    skipped = session.get('skipped_clusters', [])
    if cluster_id not in skipped:
        skipped.append(cluster_id)
    session['skipped_clusters'] = skipped
    return redirect(url_for('index'))


@app.route('/metadata_preview')
def metadata_preview():
    """Displays a planner view of pending metadata writes for review."""
    conn = get_db_connection()
    plan = [item for item in build_metadata_plan(conn) if item['requires_update']]
    ready_items = [item for item in plan if item['can_update']]
    blocked_items = [item for item in plan if not item['can_update']]
    return render_template(
        'metadata_preview.html',
        ready_items=ready_items,
        blocked_items=blocked_items,
    )


@app.route('/write_metadata', methods=['POST'])
def write_metadata():
    """Executes metadata writes for the selected planner items."""
    selected_hashes = request.form.getlist('file_hashes')
    print("Starting intelligent metadata write process...")

    conn = get_db_connection()
    plan = build_metadata_plan(conn, selected_hashes if selected_hashes else None)
    plan = [item for item in plan if item['requires_update']]

    if selected_hashes and not plan:
        flash("No metadata updates were selected for writing.", "info")
        return redirect(url_for('metadata_preview'))

    ready_items = [item for item in plan if item['can_update']]
    blocked_items = [item for item in plan if not item['can_update']]

    if not ready_items:
        if blocked_items:
            flash(
                f"No metadata updates were written. {len(blocked_items)} file(s) were unavailable.",
                "warning",
            )
        else:
            flash("No metadata updates were required.", "info")
        return redirect(url_for('metadata_preview'))

    tagged_count = 0
    failures = []

    for item in ready_items:
        video_path = item['path']
        output_path = None
        tags_string = ", ".join(item['result_people'])
        print(f"Processing {video_path} -> Tags: {tags_string}")

        try:
            input_path = video_path
            output_path = os.path.join(os.path.dirname(input_path), f".temp_{os.path.basename(input_path)}")

            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, c='copy', metadata=f"comment={item['result_comment']}")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            try:
                os.replace(output_path, input_path)
            except Exception:
                if output_path and os.path.exists(output_path):
                    os.remove(output_path)
                raise
            print("  - Successfully tagged and replaced file.")
            tagged_count += 1
        except Exception as exc:
            print(f"  - FFMPEG WRITE ERROR for file {video_path}: {exc}")
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            failures.append(item['name'] or item['file_hash'])

    if failures:
        flash(
            f"Metadata writing complete. Updated {tagged_count} file(s). Failed: {', '.join(failures)}.",
            "warning",
        )
    else:
        flash(f"Metadata writing complete. Updated {tagged_count} file(s).", "success")

    if blocked_items:
        flash(
            f"Skipped {len(blocked_items)} file(s) because their paths were unavailable.",
            "warning",
        )

    return redirect(url_for('metadata_preview'))


# --- NEW ROUTES FOR REVIEWING/EDITING ---

@app.route('/people')
def list_people():
    """Shows a grid of all identified people."""
    conn = get_db_connection()
    # Get one representative face for each person, plus a count of their faces
    people = conn.execute('''
        SELECT p.person_name, p.face_count, f.id as face_id
        FROM (
            SELECT person_name, COUNT(id) as face_count, MIN(id) as min_face_id
            FROM faces
            WHERE person_name IS NOT NULL
            GROUP BY person_name
        ) p
        JOIN faces f ON p.min_face_id = f.id
        ORDER BY p.person_name
    ''').fetchall()
    return render_template('people_list.html', people=people)


@app.route('/person/<person_name>')
def person_details(person_name):
    """Shows all faces for one person and provides editing tools."""
    conn = get_db_connection()
    faces = conn.execute('SELECT id FROM faces WHERE person_name = ?', (person_name,)).fetchall()
    if not faces:
        flash(f"Person '{person_name}' not found.", "error")
        return redirect(url_for('list_people'))
    file_rows = conn.execute('''
        SELECT DISTINCT sf.file_hash, sf.last_known_filepath
        FROM faces f
        LEFT JOIN scanned_files sf ON f.file_hash = sf.file_hash
        WHERE f.person_name = ?
        ORDER BY LOWER(COALESCE(sf.last_known_filepath, ''))
    ''', (person_name,)).fetchall()
    files = [
        {
            "file_hash": row["file_hash"],
            "path": row["last_known_filepath"],
            "name": os.path.basename(row["last_known_filepath"]) if row["last_known_filepath"] else None,
        }
        for row in file_rows
    ]
    return render_template('person_detail.html', person_name=person_name, faces=faces, files=files)


@app.route('/rename_person/<old_name>', methods=['POST'])
def rename_person(old_name):
    new_name = request.form['new_name'].strip()
    if new_name:
        conn = get_db_connection()
        conn.execute('UPDATE faces SET person_name = ? WHERE person_name = ?', (new_name, old_name))
        conn.commit()
        flash(f"Renamed '{old_name}' to '{new_name}'.", "success")
        return redirect(url_for('person_details', person_name=new_name))
    else:
        flash("New name cannot be empty.", "error")
        return redirect(url_for('person_details', person_name=old_name))


@app.route('/unname_person', methods=['POST'])
def unname_person():
    person_name = request.form['person_name']
    conn = get_db_connection()
    conn.execute('UPDATE faces SET person_name = NULL WHERE person_name = ?', (person_name,))
    conn.commit()
    flash(f"'{person_name}' has been un-named and their group is back in the queue.", "success")
    return redirect(url_for('list_people'))


@app.route('/delete_cluster_by_name', methods=['POST'])
def delete_cluster_by_name():
    person_name = request.form['person_name']
    conn = get_db_connection()
    # This is safer than deleting by cluster_id, as a person might be a result of merges
    conn.execute('DELETE FROM faces WHERE person_name = ?', (person_name,))
    conn.commit()
    flash(f"Deleted person '{person_name}' and all their face data.", "success")
    return redirect(url_for('list_people'))


@app.route('/remove_person_faces', methods=['POST'])
def remove_person_faces():
    """Deletes selected faces for a person and their thumbnails."""
    person_name = request.form['person_name']
    face_ids = request.form.getlist('face_ids')

    if not face_ids:
        flash("You must select at least one face to remove.", "error")
        return redirect(url_for('person_details', person_name=person_name))

    conn = get_db_connection()
    placeholders = ', '.join('?' for _ in face_ids)
    conn.execute(f'DELETE FROM faces WHERE id IN ({placeholders})', face_ids)
    conn.commit()

    for fid in face_ids:
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{fid}.jpg")
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

    remaining = conn.execute('SELECT 1 FROM faces WHERE person_name = ? LIMIT 1', (person_name,)).fetchone()

    flash(f"Removed {len(face_ids)} face(s).", "success")
    if remaining:
        return redirect(url_for('person_details', person_name=person_name))
    else:
        flash(f"'{person_name}' no longer has any faces and has been removed from the list.", "info")
        return redirect(url_for('list_people'))


# --- NEW ROUTES FOR MERGE/SPLIT ---

@app.route('/merge_clusters', methods=['POST'])
def merge_clusters():
    from_cluster_id = request.form['from_cluster_id']
    to_person_name = request.form.get('to_person_name')

    if not to_person_name:
        flash("You must select a person to merge with.", "error")
        return redirect(url_for('tag_group', cluster_id=from_cluster_id))

    if to_person_name.lower() == 'unknown':
        flash("Cannot merge into the Unknown person.", "error")
        return redirect(url_for('tag_group', cluster_id=from_cluster_id))

    conn = get_db_connection()
    # Find the cluster_id of the person we are merging into
    target_cluster = conn.execute('SELECT cluster_id FROM faces WHERE person_name = ? LIMIT 1',
                                  (to_person_name,)).fetchone()
    if not target_cluster:
        flash(f"Could not find target person '{to_person_name}'.", "error")
        return redirect(url_for('tag_group', cluster_id=from_cluster_id))

    to_cluster_id = target_cluster['cluster_id']

    # Reassign the old cluster_id and set the name
    conn.execute('UPDATE faces SET cluster_id = ?, person_name = ? WHERE cluster_id = ?',
                 (to_cluster_id, to_person_name, from_cluster_id))
    conn.execute('''
        UPDATE faces
        SET suggestion_status = CASE
                WHEN suggested_person_name = ? THEN 'accepted'
                ELSE 'cleared'
            END,
            suggested_person_name = NULL,
            suggested_confidence = NULL,
            suggested_candidates = NULL
        WHERE cluster_id = ?
    ''', (to_person_name, to_cluster_id))
    conn.commit()

    flash(f"Successfully merged group #{from_cluster_id} into '{to_person_name}'.", "success")
    return redirect(url_for('index'))


@app.route('/split_cluster', methods=['POST'])
def split_cluster():
    original_cluster_id = request.form['cluster_id']
    face_ids_to_split = request.form.getlist('face_ids')

    if not face_ids_to_split:
        flash("You must select at least one face to split into a new group.", "error")
        return redirect(url_for('tag_group', cluster_id=original_cluster_id))

    conn = get_db_connection()
    # Find the highest existing cluster_id to create a new one
    max_cluster_id_result = conn.execute("SELECT MAX(cluster_id) FROM faces").fetchone()
    new_cluster_id = (max_cluster_id_result[0] or 0) + 1

    # Create a placeholder string for the SQL query
    placeholders = ', '.join('?' for _ in face_ids_to_split)
    query = f"UPDATE faces SET cluster_id = ? WHERE id IN ({placeholders})"

    params = [new_cluster_id] + face_ids_to_split
    conn.execute(query, params)
    conn.commit()

    flash(
        f"Successfully split {len(face_ids_to_split)} faces from group #{original_cluster_id} into new group #{new_cluster_id}.",
        "success")
    return redirect(url_for('tag_group', cluster_id=original_cluster_id))


@app.route('/remove_faces', methods=['POST'])
def remove_faces():
    """Deletes selected faces and their thumbnails."""
    cluster_id = request.form['cluster_id']
    face_ids = request.form.getlist('face_ids')

    if not face_ids:
        flash("You must select at least one face to remove.", "error")
        return redirect(url_for('tag_group', cluster_id=cluster_id))

    conn = get_db_connection()
    remaining = _delete_faces(conn, cluster_id, face_ids)

    flash(f"Removed {len(face_ids)} face(s).", "success")
    if remaining:
        return redirect(url_for('tag_group', cluster_id=cluster_id))
    else:
        flash(f"Cluster #{cluster_id} is now empty and has been removed.", "info")
        return redirect(url_for('index'))


@app.route('/delete_selected_faces', methods=['POST'])
def delete_selected_faces():
    """Deletes selected faces entirely from the database and disk."""
    cluster_id = request.form['cluster_id']
    face_ids = request.form.getlist('face_ids')

    if not face_ids:
        flash("You must select at least one face to delete.", "error")
        return redirect(url_for('tag_group', cluster_id=cluster_id))

    conn = get_db_connection()
    remaining = _delete_faces(conn, cluster_id, face_ids)

    flash(f"Deleted {len(face_ids)} face(s).", "success")
    if remaining:
        return redirect(url_for('tag_group', cluster_id=cluster_id))
    else:
        flash(f"Cluster #{cluster_id} is now empty and has been removed.", "info")
        return redirect(url_for('index'))


@app.route('/remove_video_faces/<int:cluster_id>/<file_hash>', methods=['POST'])
def remove_video_faces(cluster_id, file_hash):
    """Removes all faces from a specific video in a cluster and creates a new cluster for them."""
    conn = get_db_connection()

    # Get all faces in the cluster that belong to the specified video
    faces_to_move = conn.execute(
        'SELECT id FROM faces WHERE cluster_id = ? AND file_hash = ?',
        (cluster_id, file_hash)
    ).fetchall()

    if not faces_to_move:
        flash("No faces found from this video in the current group.", "warning")
        return redirect(url_for('tag_group', cluster_id=cluster_id))

    # Generate a new cluster_id
    max_cluster_id_result = conn.execute("SELECT MAX(cluster_id) FROM faces").fetchone()
    new_cluster_id = (max_cluster_id_result[0] or 0) + 1

    # Move the faces to the new cluster
    face_ids = [face['id'] for face in faces_to_move]
    placeholders = ', '.join('?' for _ in face_ids)
    conn.execute(f'UPDATE faces SET cluster_id = ? WHERE id IN ({placeholders})',
                 [new_cluster_id] + face_ids)
    conn.commit()

    flash(f"Removed {len(face_ids)} face(s) from this video and created new group #{new_cluster_id}.", "success")
    return redirect(url_for('tag_group', cluster_id=cluster_id))


if __name__ == '__main__':
    # Make sure the web app is accessible from other devices on your network
    app.run(host='0.0.0.0', port=5001, debug=config.DEBUG)
