import json
import os
import re
import sqlite3
import math

import ffmpeg
from flask import (
    Flask,
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
    return {
        "unnamed_groups_count": unnamed_groups_count,
        "named_people_count": named_people_count
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
