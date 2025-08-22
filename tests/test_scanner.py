import pickle
import sqlite3
import time
from multiprocessing import Pool

import numpy as np

import scanner as scanner_module

DELAY_MAP = {}
HASH_MAP = {}


def fake_get_file_hash_with_path(filepath):
    time.sleep(DELAY_MAP.get(filepath, 0))
    return filepath, HASH_MAP.get(filepath)


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


def test_classify_new_faces_assigns_names(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    alice_encs = [np.array([0.0, 0.0]), np.array([0.1, 0.0])]
    for enc in alice_encs:
        conn.execute(
            "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name) VALUES (?, ?, ?, ?, ?)",
            ("h1", 0, "0,0,0,0", pickle.dumps(enc), "Alice"),
        )

    unknown = np.array([0.05, 0.02])
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding) VALUES (?, ?, ?, ?)",
        ("h2", 0, "0,0,0,0", pickle.dumps(unknown)),
    )
    conn.commit()

    monkeypatch.setattr(scanner_module.config, "AUTO_CLASSIFY_THRESHOLD", 0.3)
    scanner_module.classify_new_faces()

    name = conn.execute(
        "SELECT person_name FROM faces WHERE file_hash = 'h2'"
    ).fetchone()[0]
    assert name == "Alice"


def test_hashing_out_of_order_results_map_correctly(tmp_path, monkeypatch):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("a")
    f2.write_text("b")

    filepaths = [str(f1), str(f2)]

    HASH_MAP.clear()
    HASH_MAP.update({str(f1): "hash1", str(f2): "hash2"})
    DELAY_MAP.clear()
    DELAY_MAP.update({str(f1): 0.1, str(f2): 0.0})

    monkeypatch.setattr(scanner_module, "get_file_hash_with_path", fake_get_file_hash_with_path)

    hashed_files = {}
    with Pool(processes=2) as pool:
        results_iterator = pool.imap_unordered(scanner_module.get_file_hash_with_path, filepaths)
        for filepath, file_hash in results_iterator:
            hashed_files[filepath] = file_hash

    assert hashed_files == HASH_MAP
