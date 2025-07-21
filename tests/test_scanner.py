import importlib
import os
import pickle
import sqlite3

import numpy as np

import scanner as scanner_module


def setup_temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / 'faces.db'
    monkeypatch.setattr(scanner_module.config, 'DATABASE_FILE', str(db_path))
    monkeypatch.setattr(scanner_module, 'DATABASE_FILE', str(db_path))
    scanner_module.setup_database()
    return db_path


def test_setup_database_creates_tables(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'scanned_files', 'faces'} <= tables


def test_cluster_faces_updates_ids(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    # Two distinct groups of encodings
    group1 = [np.array([0.0, 0.0]), np.array([0.1, -0.1])]
    group2 = [np.array([10.0, 10.0]), np.array([10.2, 9.8])]

    for enc in group1:
        conn.execute(
            "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding) VALUES (?, ?, ?, ?)",
            ("h1", 0, "0,0,0,0", pickle.dumps(enc)),
        )
    for enc in group2:
        conn.execute(
            "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding) VALUES (?, ?, ?, ?)",
            ("h2", 0, "0,0,0,0", pickle.dumps(enc)),
        )
    conn.commit()

    monkeypatch.setattr(scanner_module.config, 'DBSCAN_EPS', 1.0)
    monkeypatch.setattr(scanner_module.config, 'DBSCAN_MIN_SAMPLES', 1)
    scanner_module.cluster_faces()

    rows = conn.execute("SELECT cluster_id FROM faces").fetchall()
    ids = {r[0] for r in rows}
    assert None not in ids
    assert len(ids) == 2
