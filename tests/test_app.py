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

