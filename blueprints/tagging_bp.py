from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, session
import os
import math
import ffmpeg

from app import get_db_connection, config


tagging_bp = Blueprint('tagging_bp', __name__)


@tagging_bp.route('/', endpoint='index')
def index():
    """Finds the next unnamed group and redirects to the tagging page for it."""
    conn = get_db_connection()
    skipped = session.get('skipped_clusters', [])

    base_query = '''
        SELECT MIN(cluster_id) as id
        FROM faces
        WHERE cluster_id IS NOT NULL AND person_name IS NULL
    '''

    params = []
    if skipped:
        placeholders = ','.join('?' for _ in skipped)
        query = base_query + f" AND cluster_id NOT IN ({placeholders})"
        params.extend(skipped)
    else:
        query = base_query

    next_group = conn.execute(query, params).fetchone()

    if (not next_group or next_group['id'] is None) and skipped:
        session.pop('skipped_clusters', None)
        next_group = conn.execute(base_query).fetchone()

    if next_group and next_group['id'] is not None:
        return redirect(url_for('tagging_bp.tag_group', cluster_id=next_group['id']))
    else:
        # No more groups to name, show a completion page
        return render_template('all_done.html')


@tagging_bp.route('/group/<int:cluster_id>', endpoint='tag_group')
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
        "SELECT id FROM faces WHERE cluster_id = ? LIMIT ? OFFSET ?",
        (cluster_id, PAGE_SIZE, offset),
    ).fetchall()

    file_rows = conn.execute('''
        SELECT DISTINCT sf.last_known_filepath
        FROM faces f
        JOIN scanned_files sf ON f.file_hash = sf.file_hash
        WHERE f.cluster_id = ?
    ''', (cluster_id,)).fetchall()
    file_names = [os.path.basename(row['last_known_filepath']) for row in file_rows]

    if not sample_faces:
        flash(f"Cluster #{cluster_id} no longer exists or is empty.", "error")
        return redirect(url_for('tagging_bp.index'))

    names = conn.execute(
        'SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL ORDER BY person_name').fetchall()

    existing_names = [name['person_name'] for name in names]
    cluster_data = {
        'id': cluster_id,
        'faces': sample_faces,
        'page': page,
        'total_pages': total_pages,
    }
    return render_template('group_tagger.html',
                           cluster=cluster_data,
                           existing_names=existing_names,
                           file_names=file_names)


@tagging_bp.route('/face_thumbnail/<int:face_id>', endpoint='get_face_thumbnail')
def get_face_thumbnail(face_id):
    """Serves a pre-generated face thumbnail."""
    thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{face_id}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype='image/jpeg')
    return "Thumbnail not found", 404


@tagging_bp.route('/name_cluster', methods=['POST'], endpoint='name_cluster')
def name_cluster():
    """Handles the form submission for naming a cluster."""
    cluster_id = request.form['cluster_id']
    person_name = request.form['person_name'].strip()

    if cluster_id and person_name:
        conn = get_db_connection()
        conn.execute('UPDATE faces SET person_name = ? WHERE cluster_id = ?', (person_name, cluster_id))
        conn.commit()
        flash(f"Assigned name '{person_name}' to cluster #{cluster_id}", "success")
    return redirect(url_for('tagging_bp.index'))


@tagging_bp.route('/delete_cluster', methods=['POST'], endpoint='delete_cluster')
def delete_cluster():
    """Deletes all data associated with a cluster_id."""
    cluster_id = request.form['cluster_id']
    if cluster_id:
        conn = get_db_connection()
        conn.execute('DELETE FROM faces WHERE cluster_id = ?', (cluster_id,))
        conn.commit()
        flash(f"Deleted all faces for cluster #{cluster_id}", "success")
    return redirect(url_for('tagging_bp.index'))


@tagging_bp.route('/skip_cluster/<int:cluster_id>', endpoint='skip_cluster')
def skip_cluster(cluster_id):
    """Marks a group as skipped so it is revisited after other groups."""
    skipped = session.get('skipped_clusters', [])
    if cluster_id not in skipped:
        skipped.append(cluster_id)
    session['skipped_clusters'] = skipped
    return redirect(url_for('tagging_bp.index'))


@tagging_bp.route('/write_metadata', methods=['POST'], endpoint='write_metadata')
def write_metadata():
    """Reads, merges, and writes face tags into the file's metadata."""
    print("Starting intelligent metadata write process...")
    conn = get_db_connection()
    videos_to_tag = conn.execute('SELECT DISTINCT file_hash FROM faces WHERE person_name IS NOT NULL').fetchall()

    tagged_count = 0
    for video in videos_to_tag:
        file_hash = video['file_hash']
        path_info = conn.execute('SELECT last_known_filepath FROM scanned_files WHERE file_hash = ?',
                                 (file_hash,)).fetchone()
        if not path_info or not os.path.exists(path_info['last_known_filepath']):
            print(f"  - WARNING: Path for hash {file_hash} not found. Skipping.")
            continue

        video_path = path_info['last_known_filepath']

        # Get names from DB
        db_names_rows = conn.execute(
            'SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL',
            (file_hash,)).fetchall()
        db_names = {row['person_name'] for row in db_names_rows}

        # Get existing names from file metadata
        try:
            probe = ffmpeg.probe(video_path)
            comment_tag = probe.get('format', {}).get('tags', {}).get('comment', '')
            if 'People: ' in comment_tag:
                existing_names_str = comment_tag.split('People: ')[1]
                existing_names = {name.strip() for name in existing_names_str.split(',')}
            else:
                existing_names = set()
        except ffmpeg.Error as e:
            print(f"  - FFPROBE ERROR for {video_path}: {e.stderr.decode('utf8')}. Assuming no existing tags.")
            existing_names = set()

        # Merge names and create new tag string
        all_names = sorted(list(db_names.union(existing_names)))
        tags_string = ", ".join(all_names)

        print(f"Processing {video_path} -> Tags: {tags_string}")

        try:
            input_path = video_path
            output_path = os.path.join(os.path.dirname(input_path), f".temp_{os.path.basename(input_path)}")

            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, c='copy', metadata=f'comment=People: {tags_string}')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            os.remove(input_path)
            os.rename(output_path, input_path)
            print(f"  - Successfully tagged and replaced file.")
            tagged_count += 1
        except Exception as e:
            print(f"  - FFMPEG WRITE ERROR for file {video_path}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)

    flash(f"Metadata writing complete. Updated {tagged_count} files.", "success")
    return redirect(url_for('tagging_bp.index'))


@tagging_bp.route('/merge_clusters', methods=['POST'], endpoint='merge_clusters')
def merge_clusters():
    from_cluster_id = request.form['from_cluster_id']
    to_person_name = request.form.get('to_person_name')

    if not to_person_name:
        flash("You must select a person to merge with.", "error")
        return redirect(url_for('tagging_bp.tag_group', cluster_id=from_cluster_id))

    conn = get_db_connection()
    target_cluster = conn.execute('SELECT cluster_id FROM faces WHERE person_name = ? LIMIT 1',
                                  (to_person_name,)).fetchone()
    if not target_cluster:
        flash(f"Could not find target person '{to_person_name}'.", "error")
        return redirect(url_for('tagging_bp.tag_group', cluster_id=from_cluster_id))

    to_cluster_id = target_cluster['cluster_id']

    conn.execute('UPDATE faces SET cluster_id = ?, person_name = ? WHERE cluster_id = ?',
                 (to_cluster_id, to_person_name, from_cluster_id))
    conn.commit()

    flash(f"Successfully merged group #{from_cluster_id} into '{to_person_name}'.", "success")
    return redirect(url_for('tagging_bp.index'))


@tagging_bp.route('/split_cluster', methods=['POST'], endpoint='split_cluster')
def split_cluster():
    original_cluster_id = request.form['cluster_id']
    face_ids_to_split = request.form.getlist('face_ids')

    if not face_ids_to_split:
        flash("You must select at least one face to split into a new group.", "error")
        return redirect(url_for('tagging_bp.tag_group', cluster_id=original_cluster_id))

    conn = get_db_connection()
    max_cluster_id_result = conn.execute("SELECT MAX(cluster_id) FROM faces").fetchone()
    new_cluster_id = (max_cluster_id_result[0] or 0) + 1

    placeholders = ', '.join('?' for _ in face_ids_to_split)
    query = f"UPDATE faces SET cluster_id = ? WHERE id IN ({placeholders})"

    params = [new_cluster_id] + face_ids_to_split
    conn.execute(query, params)
    conn.commit()

    flash(
        f"Successfully split {len(face_ids_to_split)} faces from group #{original_cluster_id} into new group #{new_cluster_id}.",
        "success")
    return redirect(url_for('tagging_bp.tag_group', cluster_id=original_cluster_id))


@tagging_bp.route('/remove_faces', methods=['POST'], endpoint='remove_faces')
def remove_faces():
    """Deletes selected faces and their thumbnails."""
    cluster_id = request.form['cluster_id']
    face_ids = request.form.getlist('face_ids')

    if not face_ids:
        flash("You must select at least one face to remove.", "error")
        return redirect(url_for('tagging_bp.tag_group', cluster_id=cluster_id))

    conn = get_db_connection()
    placeholders = ', '.join('?' for _ in face_ids)
    conn.execute(f'DELETE FROM faces WHERE id IN ({placeholders})', face_ids)
    conn.commit()

    for fid in face_ids:
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{fid}.jpg")
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

    remaining = conn.execute('SELECT 1 FROM faces WHERE cluster_id = ? LIMIT 1', (cluster_id,)).fetchone()

    flash(f"Removed {len(face_ids)} face(s).", "success")
    if remaining:
        return redirect(url_for('tagging_bp.tag_group', cluster_id=cluster_id))
    else:
        flash(f"Cluster #{cluster_id} is now empty and has been removed.", "info")
        return redirect(url_for('tagging_bp.index'))
