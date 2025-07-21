import sqlite3
import pickle

import numpy as np

import scanner as scanner_module
import app as app_module


def setup_app_db(tmp_path, monkeypatch):
    db_path = tmp_path / 'app.db'
    monkeypatch.setattr(scanner_module.config, 'DATABASE_FILE', str(db_path))
    monkeypatch.setattr(scanner_module, 'DATABASE_FILE', str(db_path))
    monkeypatch.setattr(app_module.config, 'DATABASE_FILE', str(db_path))
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
