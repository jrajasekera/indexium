from flask import Blueprint, render_template, request, redirect, url_for, flash
import os

from app import get_db_connection, config

people_bp = Blueprint('people_bp', __name__)


@people_bp.route('/people', endpoint='list_people')
def list_people():
    """Shows a grid of all identified people."""
    conn = get_db_connection()
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


@people_bp.route('/person/<person_name>', endpoint='person_details')
def person_details(person_name):
    """Shows all faces for one person and provides editing tools."""
    conn = get_db_connection()
    faces = conn.execute('SELECT id FROM faces WHERE person_name = ?', (person_name,)).fetchall()
    if not faces:
        flash(f"Person '{person_name}' not found.", "error")
        return redirect(url_for('people_bp.list_people'))
    return render_template('person_detail.html', person_name=person_name, faces=faces)


@people_bp.route('/rename_person/<old_name>', methods=['POST'], endpoint='rename_person')
def rename_person(old_name):
    new_name = request.form['new_name'].strip()
    if new_name:
        conn = get_db_connection()
        conn.execute('UPDATE faces SET person_name = ? WHERE person_name = ?', (new_name, old_name))
        conn.commit()
        flash(f"Renamed '{old_name}' to '{new_name}'.", "success")
        return redirect(url_for('people_bp.person_details', person_name=new_name))
    else:
        flash("New name cannot be empty.", "error")
        return redirect(url_for('people_bp.person_details', person_name=old_name))


@people_bp.route('/unname_person', methods=['POST'], endpoint='unname_person')
def unname_person():
    person_name = request.form['person_name']
    conn = get_db_connection()
    conn.execute('UPDATE faces SET person_name = NULL WHERE person_name = ?', (person_name,))
    conn.commit()
    flash(f"'{person_name}' has been un-named and their group is back in the queue.", "success")
    return redirect(url_for('people_bp.list_people'))


@people_bp.route('/delete_cluster_by_name', methods=['POST'], endpoint='delete_cluster_by_name')
def delete_cluster_by_name():
    person_name = request.form['person_name']
    conn = get_db_connection()
    conn.execute('DELETE FROM faces WHERE person_name = ?', (person_name,))
    conn.commit()
    flash(f"Deleted person '{person_name}' and all their face data.", "success")
    return redirect(url_for('people_bp.list_people'))


@people_bp.route('/remove_person_faces', methods=['POST'], endpoint='remove_person_faces')
def remove_person_faces():
    """Deletes selected faces for a person and their thumbnails."""
    person_name = request.form['person_name']
    face_ids = request.form.getlist('face_ids')

    if not face_ids:
        flash("You must select at least one face to remove.", "error")
        return redirect(url_for('people_bp.person_details', person_name=person_name))

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
        return redirect(url_for('people_bp.person_details', person_name=person_name))
    else:
        flash(f"'{person_name}' no longer has any faces and has been removed from the list.", "info")
        return redirect(url_for('people_bp.list_people'))
