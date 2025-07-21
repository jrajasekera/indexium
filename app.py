import os
import sqlite3
import sys

import ffmpeg
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, g

# --- CONFIGURATION ---
# Get video directory from environment variable
VIDEO_DIRECTORY = os.environ.get("INDEXIUM_VIDEO_DIR")
if VIDEO_DIRECTORY is None:
    print("Error: INDEXIUM_VIDEO_DIR environment variable not set")
    print("Please set this variable to the directory containing your videos")
    sys.exit(1)

DATABASE_FILE = "video_faces.db"
# Directory containing cached face thumbnails
THUMBNAIL_DIR = "thumbnails"

app = Flask(__name__)
# Flask needs a secret key to use flash messages
app.secret_key = os.urandom(24)

# --- DATABASE & HELPERS ---

def get_db_connection():
    """Gets a per-request database connection."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db_connection(exception):
    """Closes the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


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
    next_group = conn.execute('''
        SELECT MIN(cluster_id) as id
        FROM faces
        WHERE cluster_id IS NOT NULL AND person_name IS NULL
    ''').fetchone()

    if next_group and next_group['id'] is not None:
        return redirect(url_for('tag_group', cluster_id=next_group['id']))
    else:
        # No more groups to name, show a completion page
        return render_template('all_done.html')


@app.route('/group/<int:cluster_id>')
def tag_group(cluster_id):
    """Displays a single group for tagging."""
    conn = get_db_connection()
    sample_faces = conn.execute('SELECT id FROM faces WHERE cluster_id = ? LIMIT 50', (cluster_id,)).fetchall()

    if not sample_faces:
        flash(f"Cluster #{cluster_id} no longer exists or is empty.", "error")
        return redirect(url_for('index'))

    names = conn.execute(
        'SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL ORDER BY person_name').fetchall()

    existing_names = [name['person_name'] for name in names]
    cluster_data = {'id': cluster_id, 'faces': sample_faces}
    return render_template('group_tagger.html', cluster=cluster_data, existing_names=existing_names)


@app.route('/face_thumbnail/<int:face_id>')
def get_face_thumbnail(face_id):
    """Serves a pre-generated face thumbnail."""
    thumb_path = os.path.join(THUMBNAIL_DIR, f"{face_id}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype='image/jpeg')
    return "Thumbnail not found", 404


@app.route('/name_cluster', methods=['POST'])
def name_cluster():
    """Handles the form submission for naming a cluster."""
    cluster_id = request.form['cluster_id']
    person_name = request.form['person_name'].strip()

    if cluster_id and person_name:
        conn = get_db_connection()
        conn.execute('UPDATE faces SET person_name = ? WHERE cluster_id = ?', (person_name, cluster_id))
        conn.commit()
        flash(f"Assigned name '{person_name}' to cluster #{cluster_id}", "success")
    return redirect(url_for('index'))


@app.route('/delete_cluster', methods=['POST'])
def delete_cluster():
    """Deletes all data associated with a cluster_id."""
    cluster_id = request.form['cluster_id']
    if cluster_id:
        conn = get_db_connection()
        conn.execute('DELETE FROM faces WHERE cluster_id = ?', (cluster_id,))
        conn.commit()
        flash(f"Deleted all faces for cluster #{cluster_id}", "success")
    return redirect(url_for('index'))


@app.route('/skip_cluster/<int:cluster_id>')
def skip_cluster(cluster_id):
    """Finds the next available cluster with an ID greater than the current one."""
    conn = get_db_connection()
    next_group = conn.execute('''
        SELECT MIN(cluster_id) as id
        FROM faces
        WHERE cluster_id > ? AND cluster_id IS NOT NULL AND person_name IS NULL
    ''', (cluster_id,)).fetchone()

    if next_group and next_group['id'] is not None:
        return redirect(url_for('tag_group', cluster_id=next_group['id']))
    else:
        flash("No more groups to skip to. Looping back to the start.", "success")
        return redirect(url_for('index'))


@app.route('/write_metadata', methods=['POST'])
def write_metadata():
    """
    Intelligently reads, merges, and writes face tags into the file's
    metadata 'comment' tag using ffmpeg.
    """
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
            if os.path.exists(output_path): os.remove(output_path)

    flash(f"Metadata writing complete. Updated {tagged_count} files.", "success")
    return redirect(url_for('index'))


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
    return render_template('person_detail.html', person_name=person_name, faces=faces)


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


# --- NEW ROUTES FOR MERGE/SPLIT ---

@app.route('/merge_clusters', methods=['POST'])
def merge_clusters():
    from_cluster_id = request.form['from_cluster_id']
    to_person_name = request.form.get('to_person_name')

    if not to_person_name:
        flash("You must select a person to merge with.", "error")
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


if __name__ == '__main__':
    # Make sure the web app is accessible from other devices on your network
    app.run(host='0.0.0.0', port=5001, debug=True)
