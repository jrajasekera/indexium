import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np

import app as app_module
import scanner as scanner_module


def setup_app_db(tmp_path, monkeypatch):
    db_path = tmp_path / 'app.db'
    monkeypatch.setattr(scanner_module.config, 'DATABASE_FILE', str(db_path))
    monkeypatch.setattr(scanner_module, 'DATABASE_FILE', str(db_path))
    monkeypatch.setattr(app_module.config, 'DATABASE_FILE', str(db_path))
    thumb_dir = tmp_path / 'thumbs'
    monkeypatch.setattr(app_module.config, 'THUMBNAIL_DIR', str(thumb_dir))
    monkeypatch.setattr(scanner_module.config, 'THUMBNAIL_DIR', str(thumb_dir))
    scanner_module.setup_database()
    return db_path


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
