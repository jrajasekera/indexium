import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np

import app as app_module
import scanner as scanner_module
from text_utils import calculate_top_text_fragments


def setup_app_db(tmp_path, monkeypatch):
    db_path = tmp_path / 'app.db'
    monkeypatch.setattr(scanner_module.config, 'DATABASE_FILE', str(db_path))
    monkeypatch.setattr(scanner_module, 'DATABASE_FILE', str(db_path))
    monkeypatch.setattr(app_module.config, 'DATABASE_FILE', str(db_path))
    thumb_dir = tmp_path / 'thumbs'
    monkeypatch.setattr(app_module.config, 'THUMBNAIL_DIR', str(thumb_dir))
    monkeypatch.setattr(scanner_module.config, 'THUMBNAIL_DIR', str(thumb_dir))
    sample_dir = thumb_dir / 'no_faces'
    monkeypatch.setattr(app_module.config, 'NO_FACE_SAMPLE_DIR', str(sample_dir))
    monkeypatch.setattr(scanner_module.config, 'NO_FACE_SAMPLE_DIR', str(sample_dir))
    scanner_module.setup_database()
    return db_path


def test_calculate_top_text_fragments_prefers_frequent_longer_substrings():
    entries = [
        ("Hello World", 1),
        ("Hello There", 1),
        ("World Hello", 2),
    ]

    results = calculate_top_text_fragments(entries, top_n=5)
    assert results
    assert results[0]['substring'].strip().lower() == 'hello'
    assert results[0]['count'] == 4
    assert all(len(item['substring']) >= 4 for item in results)


def test_skip_cluster_route(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    with app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['skipped_clusters'] = []
        resp = client.get('/skip_cluster/5')
        assert resp.status_code == 302
        with client.session_transaction() as sess:
            assert 5 in sess['skipped_clusters']


def test_get_progress_stats(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    # cluster 1 unnamed
    enc = pickle.dumps(np.array([0]))
    conn.execute("INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
                 ('h1', 0, '0,0,0,0', enc, 1))
    # cluster 2 named Alice
    conn.execute("INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
                 ('h2', 0, '0,0,0,0', enc, 2, 'Alice'))
    # cluster 3 named Bob
    conn.execute("INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
                 ('h3', 0, '0,0,0,0', enc, 3, 'Bob'))
    conn.commit()

    with app_module.app.app_context():
        stats = app_module.get_progress_stats()
    assert stats['unnamed_groups_count'] == 1
    assert stats['named_people_count'] == 2
    assert stats['manual_pending_videos'] == 0
    assert stats['manual_completed_videos'] == 0


def test_remove_faces_route(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        ("h1", 1, "0,0,0,0", enc, 1),
    )
    conn.commit()
    ids = [row[0] for row in conn.execute("SELECT id FROM faces").fetchall()]
    remove_id = ids[0]
    thumb_dir = Path(app_module.config.THUMBNAIL_DIR)
    thumb_dir.mkdir()
    (thumb_dir / f"{remove_id}.jpg").write_bytes(b"test")

    with app_module.app.test_client() as client:
        resp = client.post(
            "/remove_faces",
            data={"cluster_id": 1, "face_ids": [str(remove_id)]},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    conn = sqlite3.connect(db_path)
    remaining = [row[0] for row in conn.execute("SELECT id FROM faces").fetchall()]
    assert remove_id not in remaining
    assert not (thumb_dir / f"{remove_id}.jpg").exists()


def test_delete_selected_faces_route(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        ("h1", 1, "0,0,0,0", enc, 1),
    )
    conn.commit()
    ids = [row[0] for row in conn.execute("SELECT id FROM faces").fetchall()]
    delete_id = ids[0]
    thumb_dir = Path(app_module.config.THUMBNAIL_DIR)
    thumb_dir.mkdir()
    (thumb_dir / f"{delete_id}.jpg").write_bytes(b"test")

    with app_module.app.test_client() as client:
        resp = client.post(
            "/delete_selected_faces",
            data={"cluster_id": 1, "face_ids": [str(delete_id)]},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    conn = sqlite3.connect(db_path)
    remaining = [row[0] for row in conn.execute("SELECT id FROM faces").fetchall()]
    assert delete_id not in remaining
    assert not (thumb_dir / f"{delete_id}.jpg").exists()


def test_manual_detail_autofills_exact_match(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    file_hash = 'exact123'
    video_path = tmp_path / 'Alice Johnson.mp4'
    video_path.write_bytes(b'')

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, manual_review_status, face_count, ocr_text_count) VALUES (?, ?, ?, ?, ?)",
        (file_hash, str(video_path), 'pending', 0, 0),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        (file_hash, 0, '0,0,0,0', enc, 1, 'Alice Johnson'),
    )
    conn.commit()

    monkeypatch.setattr(app_module, '_get_video_samples', lambda *args, **kwargs: ([], str(video_path)))

    with app_module.app.test_client() as client:
        resp = client.get(f'/videos/manual/{file_hash}')
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

    assert 'value="Alice Johnson"' in html
    assert 'Suggested: Alice Johnson' in html


def test_manual_detail_autofills_fuzzy_match_from_ocr(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    file_hash = 'fuzzy123'
    video_path = tmp_path / 'conference_clip.mp4'
    video_path.write_bytes(b'')

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, manual_review_status, face_count, ocr_text_count) VALUES (?, ?, ?, ?, ?)",
        (file_hash, str(video_path), 'pending', 0, 1),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        (file_hash, 0, '0,0,0,0', enc, 2, 'Bob Smith'),
    )
    conn.execute(
        "INSERT INTO video_text (file_hash, raw_text, normalized_text, confidence, first_seen_frame, first_seen_timestamp_ms, occurrence_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (file_hash, 'Keynote with Bob Smiht', 'keynote with bob smiht', 0.9, 0, 0.0, 1),
    )
    conn.commit()

    monkeypatch.setattr(app_module, '_get_video_samples', lambda *args, **kwargs: ([], str(video_path)))

    with app_module.app.test_client() as client:
        resp = client.get(f'/videos/manual/{file_hash}')
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

    assert 'value="Bob Smith"' in html
    assert 'Suggested: Bob Smith' in html


def test_metadata_plan_api_returns_json(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    video_path = tmp_path / 'api_video.mp4'
    video_path.write_text('video')

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ('api1', str(video_path)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ('api1', 0, '0,0,0,0', enc, 10, 'Dana'),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(app_module.ffmpeg, 'probe', lambda path: {'format': {'tags': {'comment': 'People: Dana'}}})

    with app_module.app.test_client() as client:
        resp = client.get('/api/metadata/plan')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['statistics']['total_files'] == 1
        assert data['items'][0]['file_hash'] == 'api1'
        assert data['items'][0]['risk_level'] == 'safe'
        assert data['filters']['file_types'] == ['.mp4']
        assert data['insights']['total_people'] == 1


def test_metadata_plan_edit_endpoint_allows_custom_people(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    video_path = tmp_path / 'api_video_edit.mp4'
    video_path.write_text('video')

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ('api2', str(video_path)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ('api2', 0, '0,0,0,0', enc, 11, 'Evan'),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(app_module.ffmpeg, 'probe', lambda path: {'format': {'tags': {'comment': ''}}})

    with app_module.app.test_client() as client:
        resp = client.post('/api/metadata/plan/api2/edit', json={'result_people': ['Evan', 'Fran']})
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload['item']['result_people'] == ['Evan', 'Fran']
        assert set(payload['item']['tags_to_add']) == {'Evan', 'Fran'}


def test_metadata_plan_api_supports_filters_and_sort(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video_one = tmp_path / 'sorted_a.mp4'
    video_one.write_text('video1')
    video_two = tmp_path / 'sorted_b.mov'
    video_two.write_text('video2')

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ('s1', str(video_one)),
    )
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ('s2', str(video_two)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ('s1', 0, '0,0,0,0', enc, 20, 'Gina'),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ('s2', 0, '0,0,0,0', enc, 21, 'Hank'),
    )
    conn.commit()
    conn.close()

    def fake_probe(path):
        comment_map = {
            str(video_one): 'People: Gina',
            str(video_two): '',
        }
        return {'format': {'tags': {'comment': comment_map.get(path, '')}}}

    monkeypatch.setattr(app_module.ffmpeg, 'probe', fake_probe)

    with app_module.app.test_client() as client:
        resp = client.post(
            '/api/metadata/plan',
            json={
                'filter': {'file_types': ['mp4']},
                'sort': {'by': 'tag_count', 'direction': 'desc'},
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['items']) == 1
        assert data['items'][0]['file_hash'] == 's1'

        resp_missing = client.post(
            '/api/metadata/plan',
            json={'filter': {'issue_codes': ['missing_file']}},
        )
        assert resp_missing.status_code == 200
        data_missing = resp_missing.get_json()
        assert isinstance(data_missing['items'], list)


def test_mark_unknown_route(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, suggested_person_name, suggested_confidence, suggestion_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Alice", 0.8, "pending"),
    )
    conn.commit()

    with app_module.app.test_client() as client:
        resp = client.post(
            "/mark_unknown",
            data={"cluster_id": "1"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT cluster_id, person_name, suggested_person_name, suggested_confidence, suggestion_status FROM faces"
    ).fetchone()
    assert row[0] == -1
    assert row[1] == "Unknown"
    assert row[2] is None
    assert row[3] is None
    assert row[4] is None


def test_remove_person_faces_route(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name) VALUES (?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, "Alice"),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name) VALUES (?, ?, ?, ?, ?)",
        ("h1", 1, "0,0,0,0", enc, "Alice"),
    )
    conn.commit()
    ids = [row[0] for row in conn.execute("SELECT id FROM faces").fetchall()]
    remove_id = ids[0]
    thumb_dir = Path(app_module.config.THUMBNAIL_DIR)
    thumb_dir.mkdir()
    (thumb_dir / f"{remove_id}.jpg").write_bytes(b"test")

    with app_module.app.test_client() as client:
        resp = client.post(
            "/remove_person_faces",
            data={"person_name": "Alice", "face_ids": [str(remove_id)]},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    conn = sqlite3.connect(db_path)
    remaining = [row[0] for row in conn.execute("SELECT id FROM faces").fetchall()]
    assert remove_id not in remaining
    assert len(remaining) == 1
    assert not (thumb_dir / f"{remove_id}.jpg").exists()


def test_tag_group_pagination(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    for i in range(60):
        conn.execute(
            "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
            ("h1", i, "0,0,0,0", enc, 1),
        )
    conn.commit()

    with app_module.app.test_client() as client:
        resp1 = client.get("/group/1?page=1")
        assert resp1.status_code == 200
        assert b"Page 1 of 2" in resp1.data

        resp2 = client.get("/group/1?page=2")
        assert resp2.status_code == 200
        assert b"Page 2 of 2" in resp2.data


def test_manual_video_tagging_workflow(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    file_hash = 'vh1'
    video_path = tmp_path / 'missing.mp4'
    conn.execute(
        "INSERT OR REPLACE INTO scanned_files (file_hash, last_known_filepath, manual_review_status, face_count) VALUES (?, ?, ?, ?)",
        (file_hash, str(video_path), 'pending', 0),
    )
    conn.commit()

    with app_module.app.test_client() as client:
        dashboard = client.get('/videos/manual')
        assert dashboard.status_code == 200

        resp = client.get(f'/videos/manual/{file_hash}')
        assert resp.status_code == 200

        conn = sqlite3.connect(db_path)
        status = conn.execute(
            'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
            (file_hash,),
        ).fetchone()[0]
        assert status == 'in_progress'

        # Cannot mark done without tags
        resp = client.post(
            f'/videos/manual/{file_hash}/status',
            data={'status': 'done'},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        conn = sqlite3.connect(db_path)
        status = conn.execute(
            'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
            (file_hash,),
        ).fetchone()[0]
        assert status == 'in_progress'

        # Add a tag
        resp = client.post(
            f'/videos/manual/{file_hash}/tags',
            data={'person_name': 'Alice'},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        conn = sqlite3.connect(db_path)
        tags = conn.execute(
            'SELECT person_name FROM video_people WHERE file_hash = ?',
            (file_hash,),
        ).fetchall()
        assert [row[0] for row in tags] == ['Alice']

        # Remove the tag
        resp = client.post(
            f'/videos/manual/{file_hash}/tags/remove',
            data={'person_name': ['Alice']},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        conn = sqlite3.connect(db_path)
        count = conn.execute(
            'SELECT COUNT(*) FROM video_people WHERE file_hash = ?',
            (file_hash,),
        ).fetchone()[0]
        assert count == 0

        # Add another tag and mark as done in one step
        resp = client.post(
            f'/videos/manual/{file_hash}/tags',
            data={'person_name': 'Bob', 'submit_action': 'add_and_done'},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers['Location'].endswith('/videos/manual')
        conn = sqlite3.connect(db_path)
        status = conn.execute(
            'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
            (file_hash,),
        ).fetchone()[0]
        assert status == 'done'

        # Mark as no people clears tags
        resp = client.post(
            f'/videos/manual/{file_hash}/status',
            data={'status': 'no_people'},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        conn = sqlite3.connect(db_path)
        status = conn.execute(
            'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
            (file_hash,),
        ).fetchone()[0]
        count = conn.execute(
            'SELECT COUNT(*) FROM video_people WHERE file_hash = ?',
            (file_hash,),
        ).fetchone()[0]
        assert status == 'no_people'
        assert count == 0

    resp = client.get('/videos/manual?person=Bob')
    assert resp.status_code == 200
    assert b'No manual videos currently reference Bob' in resp.data


def test_manual_status_redirects_to_next_video(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    file_hash_one = 'vh1'
    file_hash_two = 'vh2'

    conn.execute(
        "INSERT OR REPLACE INTO scanned_files (file_hash, last_known_filepath, manual_review_status, face_count) VALUES (?, ?, ?, ?)",
        (file_hash_one, str(tmp_path / 'missing1.mp4'), 'pending', 0),
    )
    conn.execute(
        "INSERT OR REPLACE INTO scanned_files (file_hash, last_known_filepath, manual_review_status, face_count) VALUES (?, ?, ?, ?)",
        (file_hash_two, str(tmp_path / 'missing2.mp4'), 'pending', 0),
    )
    conn.execute(
        "INSERT INTO video_people (file_hash, person_name) VALUES (?, ?)",
        (file_hash_one, 'Alice'),
    )
    conn.commit()

    with app_module.app.test_client() as client:
        resp = client.post(
            f'/videos/manual/{file_hash_one}/status',
            data={'status': 'done'},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers['Location'].endswith(f'/videos/manual/{file_hash_two}')

    conn = sqlite3.connect(db_path)
    status_one = conn.execute(
        'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
        (file_hash_one,),
    ).fetchone()[0]
    status_two = conn.execute(
        'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
        (file_hash_two,),
    ).fetchone()[0]
    assert status_one == 'done'
    assert status_two == 'pending'


def test_manual_add_and_done_redirects_to_next_video(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    file_hash_one = 'vh1'
    file_hash_two = 'vh2'

    conn.execute(
        "INSERT OR REPLACE INTO scanned_files (file_hash, last_known_filepath, manual_review_status, face_count) VALUES (?, ?, ?, ?)",
        (file_hash_one, str(tmp_path / 'missing1.mp4'), 'pending', 0),
    )
    conn.execute(
        "INSERT OR REPLACE INTO scanned_files (file_hash, last_known_filepath, manual_review_status, face_count) VALUES (?, ?, ?, ?)",
        (file_hash_two, str(tmp_path / 'missing2.mp4'), 'pending', 0),
    )
    conn.commit()

    with app_module.app.test_client() as client:
        resp = client.get(f'/videos/manual/{file_hash_one}')
        assert resp.status_code == 200

        resp = client.post(
            f'/videos/manual/{file_hash_one}/tags',
            data={'person_name': 'Alice', 'submit_action': 'add_and_done'},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers['Location'].endswith(f'/videos/manual/{file_hash_two}')

    conn = sqlite3.connect(db_path)
    status_one = conn.execute(
        'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
        (file_hash_one,),
    ).fetchone()[0]
    status_two = conn.execute(
        'SELECT manual_review_status FROM scanned_files WHERE file_hash = ?',
        (file_hash_two,),
    ).fetchone()[0]
    tags = conn.execute(
        'SELECT person_name FROM video_people WHERE file_hash = ?',
        (file_hash_one,),
    ).fetchall()

    assert status_one == 'done'
    assert status_two == 'pending'
    assert [row[0] for row in tags] == ['Alice']


def test_tag_group_select_buttons(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1),
    )
    conn.commit()

    with app_module.app.test_client() as client:
        resp = client.get("/group/1")
        assert resp.status_code == 200
        assert b"Select All Faces" in resp.data
        assert b"Unselect All Faces" in resp.data


def test_accept_suggestion_route(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, suggested_person_name, suggested_confidence, suggestion_status, suggested_candidates) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Alice", 0.9, "pending", json.dumps([
            {"name": "Alice", "confidence": 0.9},
            {"name": "Bob", "confidence": 0.4}
        ])),
    )
    conn.commit()

    with app_module.app.test_client() as client:
        resp = client.post(
            "/accept_suggestion",
            data={"cluster_id": "1", "suggestion_name": "Alice"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    conn.close()
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT person_name, suggested_person_name, suggestion_status, suggested_candidates FROM faces WHERE cluster_id = 1"
    ).fetchone()
    assert row[0] == "Alice"
    assert row[1] is None
    assert row[2] == "accepted"
    assert row[3] is None


def test_reject_suggestion_route(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, suggested_person_name, suggested_confidence, suggestion_status, suggested_candidates) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Bob", 0.7, "pending", json.dumps([
            {"name": "Bob", "confidence": 0.7},
            {"name": "Alice", "confidence": 0.5}
        ])),
    )
    conn.commit()

    with app_module.app.test_client() as client:
        resp = client.post(
            "/reject_suggestion",
            data={"cluster_id": "1", "suggestion_name": "Bob"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    conn.close()
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT person_name, suggested_person_name, suggestion_status, suggested_candidates FROM faces WHERE cluster_id = 1"
    ).fetchone()
    assert row[0] is None
    assert row[1] == "Bob"
    assert row[2] == "rejected"
    assert row[3] is not None


def test_tag_group_shows_suggestion(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, suggested_person_name, suggested_confidence, suggestion_status, suggested_candidates) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Carol", 0.8, "pending", json.dumps([
            {"name": "Carol", "confidence": 0.82},
            {"name": "Alice", "confidence": 0.3},
            {"name": "Bob", "confidence": 0.28}
        ])),
    )
    conn.commit()

    conn.close()

    with app_module.app.test_client() as client:
        resp = client.get("/group/1")
        assert resp.status_code == 200
        assert b"Suggested match" in resp.data
        assert b"Carol" in resp.data
        assert b"Top Matches" in resp.data

def test_write_metadata_preserves_file_on_failure(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    video_path = tmp_path / "video.mp4"
    video_path.write_text("original")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h1", str(video_path)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Alice"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(app_module.ffmpeg, "probe", lambda path: {"format": {"tags": {"comment": ""}}})

    def fake_run(stream, overwrite_output=True, quiet=True):
        temp = video_path.parent / f".temp_{video_path.name}"
        temp.write_text("temp")

    monkeypatch.setattr(app_module.ffmpeg, "run", fake_run)

    def fake_replace(src, dst):
        raise OSError("replace fail")

    monkeypatch.setattr(app_module.os, "replace", fake_replace)

    with app_module.app.test_client() as client:
        resp = client.post("/write_metadata", follow_redirects=False)
        assert resp.status_code == 302

    assert video_path.exists()
    assert video_path.read_text() == "original"
    temp_path = video_path.parent / f".temp_{video_path.name}"
    assert not temp_path.exists()


def test_metadata_preview_lists_pending_updates(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    video_path = tmp_path / "video1.mp4"
    video_path.write_text("original")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h1", str(video_path)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Alice"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(app_module.ffmpeg, "probe", lambda path: {"format": {"tags": {"comment": ""}}})

    with app_module.app.test_client() as client:
        resp = client.get("/metadata_preview")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Smart Metadata Planner" in html
        assert 'id="planner-app"' in html
        assert 'data-component="planner-table"' in html


def test_write_metadata_respects_selection(tmp_path, monkeypatch):
    db_path = setup_app_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video_one = tmp_path / "video1.mp4"
    video_one.write_text("video1")
    video_two = tmp_path / "video2.mp4"
    video_two.write_text("video2")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h1", str(video_one)),
    )
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h2", str(video_two)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Alice"),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("h2", 0, "0,0,0,0", enc, 2, "Bob"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(app_module.ffmpeg, "probe", lambda path: {"format": {"tags": {"comment": ""}}})
    class DummyWriter:
        def __init__(self):
            self.calls = []

        def start_operation(self, items, options, background=True):
            self.calls.append((items, options, background))
            return 123

        def get_operation_status(self, operation_id):
            return {'status': 'completed', 'items': []}

        def pause_operation(self, operation_id):
            return False

        def resume_operation(self, operation_id):
            return False

        def cancel_operation(self, operation_id):
            return False

    dummy_writer = DummyWriter()
    monkeypatch.setattr(app_module, 'metadata_writer', dummy_writer)

    with app_module.app.test_client() as client:
        resp = client.post(
            "/write_metadata",
            data={"file_hashes": ["h1"]},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers['Location'].endswith('/metadata_progress?operation_id=123')

    assert len(dummy_writer.calls) == 1
    call_items, call_options, background = dummy_writer.calls[0]
    assert background is True
    assert len(call_items) == 1
    assert call_items[0].file_hash == 'h1'


def test_write_metadata_requires_selection(tmp_path, monkeypatch):
    setup_app_db(tmp_path, monkeypatch)
    with app_module.app.test_client() as client:
        resp = client.post(
            "/write_metadata",
            data={},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers['Location'].endswith('/metadata_preview')


def test_metadata_progress_route_handles_missing_operation(tmp_path, monkeypatch):
    setup_app_db(tmp_path, monkeypatch)

    class DummyWriter:
        def get_operation_status(self, operation_id):
            return None

    monkeypatch.setattr(app_module, 'metadata_writer', DummyWriter())

    with app_module.app.test_client() as client:
        resp = client.get('/metadata_progress?operation_id=999', follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers['Location'].endswith('/metadata_preview')


def test_metadata_operation_status_endpoint(tmp_path, monkeypatch):
    setup_app_db(tmp_path, monkeypatch)

    sample_status = {
        'operation_id': 5,
        'status': 'in_progress',
        'file_count': 2,
        'success_count': 1,
        'failure_count': 0,
        'pending_count': 1,
        'skipped_count': 0,
        'items': [
            {
                'id': 1,
                'file_hash': 'abc',
                'file_path': '/tmp/a.mp4',
                'file_name': 'a.mp4',
                'status': 'success',
                'error_message': None,
                'processed_at': None,
                'tags_added': ['Alice'],
                'tags_removed': [],
            }
        ],
    }

    class DummyWriter:
        def get_operation_status(self, operation_id):
            if operation_id == 5:
                return sample_status
            return None

        def pause_operation(self, operation_id):
            return True

        def resume_operation(self, operation_id):
            return True

        def cancel_operation(self, operation_id):
            return True

    dummy_writer = DummyWriter()
    monkeypatch.setattr(app_module, 'metadata_writer', dummy_writer)

    with app_module.app.test_client() as client:
        resp = client.get('/api/metadata/operations/5')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'in_progress'
        assert data['file_count'] == 2

        resp = client.post('/api/metadata/operations/5/pause')
        assert resp.status_code == 200

        resp = client.post('/api/metadata/operations/5/resume')
        assert resp.status_code == 200

        resp = client.post('/api/metadata/operations/5/cancel')
        assert resp.status_code == 200
