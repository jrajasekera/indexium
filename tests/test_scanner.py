import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import scanner as scanner_module
from text_utils import calculate_top_text_fragments

DELAY_MAP = {}
HASH_MAP = {}


def fake_get_file_hash_with_path(filepath):
    time.sleep(DELAY_MAP.get(filepath, 0))
    return filepath, HASH_MAP.get(filepath)


def setup_temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "faces.db"
    monkeypatch.setattr(scanner_module.config, "DATABASE_FILE", str(db_path))
    monkeypatch.setattr(scanner_module, "DATABASE_FILE", str(db_path))
    scanner_module.setup_database()
    monkeypatch.setattr(scanner_module, "ACTIVE_OCR_BACKEND", None, raising=False)
    monkeypatch.setattr(scanner_module, "OCR_ENABLED", True, raising=False)
    monkeypatch.setattr(scanner_module, "_ocr_reader", None, raising=False)
    return db_path


def test_collect_ocr_from_frame_merges_text(monkeypatch):
    class DummyReader:
        def __init__(self, payload):
            self.payload = payload

        def readtext(self, frame, detail=1):  # noqa: D401 - signature match
            return self.payload

    aggregator = {}
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    primary_payload = [
        (None, "Hello World", 0.8),
        (None, " hello   world ", 0.6),
        (None, "ignored", 0.3),
    ]

    secondary_payload = [
        (None, "Different", 0.9),
        (None, "Hello World", 0.7),
    ]

    payloads = [primary_payload, secondary_payload]
    call_index = {"value": 0}

    def fake_request(frame):
        idx = call_index["value"]
        call_index["value"] += 1
        return payloads[min(idx, len(payloads) - 1)]

    def fake_init():
        scanner_module.ACTIVE_OCR_BACKEND = "easyocr"

    monkeypatch.setattr(scanner_module, "OCR_ENABLED", True, raising=False)
    monkeypatch.setattr(scanner_module, "OCR_MIN_CONFIDENCE", 0.5, raising=False)
    monkeypatch.setattr(scanner_module, "OCR_MAX_RESULTS_PER_VIDEO", 5, raising=False)
    monkeypatch.setattr(scanner_module, "initialize_ocr_backend", fake_init, raising=False)
    monkeypatch.setattr(scanner_module, "_easyocr_request", fake_request, raising=False)
    monkeypatch.setattr(scanner_module, "_BACKEND_INITIALIZED", True, raising=False)
    monkeypatch.setattr(scanner_module, "ACTIVE_OCR_BACKEND", "easyocr", raising=False)

    scanner_module.collect_ocr_from_frame(frame, frame_index=30, fps=30.0, aggregator=aggregator)
    scanner_module.collect_ocr_from_frame(frame, frame_index=60, fps=30.0, aggregator=aggregator)

    entries = scanner_module.serialize_ocr_entries("hash", aggregator)
    assert len(entries) == 2

    hello_entry = next(item for item in entries if item["normalized_text"] == "hello world")
    assert hello_entry["occurrence_count"] == 3
    assert hello_entry["confidence"] == 0.8  # highest confidence retained
    assert hello_entry["first_seen_frame"] == 30

    different_entry = next(item for item in entries if item["normalized_text"] == "different")
    assert different_entry["occurrence_count"] == 1
    assert different_entry["confidence"] == 0.9


def test_setup_database_creates_tables(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"scanned_files", "faces", "video_text"} <= tables

    columns = {row[1] for row in conn.execute("PRAGMA table_info(scanned_files)")}
    assert {"ocr_text_count", "ocr_last_updated"} <= columns


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

    monkeypatch.setattr(scanner_module.config, "DBSCAN_EPS", 1.0)
    monkeypatch.setattr(scanner_module.config, "DBSCAN_MIN_SAMPLES", 1)
    scanner_module.cluster_faces()

    rows = conn.execute("SELECT cluster_id FROM faces").fetchall()
    ids = {r[0] for r in rows}
    assert None not in ids
    assert len(ids) == 2


def test_write_data_to_db_persists_ocr(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")

    ocr_payload = [
        {
            "raw_text": "Hello World",
            "normalized_text": "hello world",
            "confidence": 0.87,
            "first_seen_frame": 42,
            "first_seen_timestamp_ms": 1400,
            "occurrence_count": 3,
        },
        {
            "raw_text": "Indexium",
            "normalized_text": "indexium",
            "confidence": 0.65,
            "first_seen_frame": 60,
            "first_seen_timestamp_ms": 2000,
            "occurrence_count": 1,
        },
    ]

    fragments = calculate_top_text_fragments(
        [(item["raw_text"], item["occurrence_count"]) for item in ocr_payload],
        scanner_module.TOP_FRAGMENT_COUNT,
        scanner_module.MIN_OCR_TEXT_LENGTH,
    )

    scanner_module.write_data_to_db(
        face_data=[],
        scanned_files_info=[("hash1", str(video_path), 0, len(ocr_payload))],
        failed_files_info=None,
        ocr_text_data=[("hash1", ocr_payload)],
        ocr_fragments_data=[("hash1", fragments)],
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT raw_text, normalized_text, confidence, occurrence_count FROM video_text WHERE file_hash = ?",
        ("hash1",),
    ).fetchall()
    assert len(rows) == 2
    assert {row[1] for row in rows} == {"hello world", "indexium"}

    fragment_rows = conn.execute(
        "SELECT fragment_text, occurrence_count FROM video_text_fragments WHERE file_hash = ? ORDER BY rank",
        ("hash1",),
    ).fetchall()
    assert fragment_rows

    ocr_meta = conn.execute(
        "SELECT ocr_text_count, ocr_last_updated, manual_review_status FROM scanned_files WHERE file_hash = ?",
        ("hash1",),
    ).fetchone()
    assert ocr_meta[0] == 2
    assert ocr_meta[1] is not None
    assert ocr_meta[2] == "pending"


def test_refresh_ocr_data_updates_existing_rows(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO scanned_files (
            file_hash,
            last_known_filepath,
            processing_status,
            face_count,
            manual_review_status,
            ocr_text_count
        ) VALUES (?, ?, 'completed', ?, 'not_required', 0)
        """,
        ("hash-refresh", str(video_path), 2),
    )
    conn.commit()

    sample_entries = [
        {
            "raw_text": "Refresh Me",
            "normalized_text": "refresh me",
            "confidence": 0.9,
            "first_seen_frame": 10,
            "first_seen_timestamp_ms": 400,
            "occurrence_count": 1,
        }
    ]

    sample_fragments = calculate_top_text_fragments(
        [(item["raw_text"], item["occurrence_count"]) for item in sample_entries],
        scanner_module.TOP_FRAGMENT_COUNT,
        scanner_module.MIN_OCR_TEXT_LENGTH,
    )

    def fake_process(job):
        file_path, file_hash = job
        return (file_hash, file_path, True, [], sample_entries, sample_fragments, None)

    def fake_init():
        scanner_module.ACTIVE_OCR_BACKEND = "easyocr"

    monkeypatch.setattr(scanner_module, "OCR_ENABLED", True, raising=False)
    monkeypatch.setattr(scanner_module, "initialize_ocr_backend", fake_init, raising=False)
    monkeypatch.setattr(scanner_module, "process_video_job", fake_process)
    monkeypatch.setattr(scanner_module, "pytesseract", None, raising=False)

    scanner_module.refresh_ocr_data(["hash-refresh"])

    rows = conn.execute(
        "SELECT raw_text, normalized_text FROM video_text WHERE file_hash = ?",
        ("hash-refresh",),
    ).fetchall()
    assert rows == [("Refresh Me", "refresh me")]

    fragment_rows = conn.execute(
        "SELECT fragment_text FROM video_text_fragments WHERE file_hash = ?",
        ("hash-refresh",),
    ).fetchall()
    assert fragment_rows


def test_continue_ocr_data_skips_completed(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    video1 = tmp_path / "vid1.mp4"
    video2 = tmp_path / "vid2.mp4"
    video1.write_bytes(b"vid1")
    video2.write_bytes(b"vid2")

    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO scanned_files (
            file_hash,
            last_known_filepath,
            processing_status,
            face_count,
            manual_review_status,
            ocr_text_count
        ) VALUES (?, ?, 'completed', ?, 'not_required', ?)
        """,
        [
            ("hash1", str(video1), 0, 0),
            ("hash2", str(video2), 0, 2),
        ],
    )
    conn.execute(
        "INSERT INTO video_text (file_hash, raw_text, normalized_text, occurrence_count) VALUES (?, ?, ?, ?)",
        ("hash2", "existing", "existing", 1),
    )
    conn.commit()

    sample_entries = [
        {
            "raw_text": "Only Once",
            "normalized_text": "only once",
            "confidence": 0.9,
            "first_seen_frame": 0,
            "first_seen_timestamp_ms": 0,
            "occurrence_count": 1,
        }
    ]
    sample_fragments = calculate_top_text_fragments(
        [(item["raw_text"], item["occurrence_count"]) for item in sample_entries],
        scanner_module.TOP_FRAGMENT_COUNT,
        scanner_module.MIN_OCR_TEXT_LENGTH,
    )

    processed = []

    def fake_process(job):
        processed.append(job)
        file_path, file_hash = job
        return (file_hash, file_path, True, [], sample_entries, sample_fragments, None)

    def fake_init():
        scanner_module.ACTIVE_OCR_BACKEND = "easyocr"

    monkeypatch.setattr(scanner_module, "OCR_ENABLED", True, raising=False)
    monkeypatch.setattr(scanner_module, "initialize_ocr_backend", fake_init, raising=False)
    monkeypatch.setattr(scanner_module, "process_video_job", fake_process)
    monkeypatch.setattr(scanner_module, "pytesseract", None, raising=False)

    scanner_module.continue_ocr_data()

    assert processed == [(str(video1), "hash1")]

    rows = conn.execute("SELECT raw_text FROM video_text WHERE file_hash = 'hash1'").fetchall()
    assert rows
    ocr_meta = conn.execute(
        "SELECT ocr_text_count FROM scanned_files WHERE file_hash = ?",
        ("hash1",),
    ).fetchone()
    assert ocr_meta[0] == 1


def test_cleanup_ocr_text_removes_short_entries(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, processing_status, ocr_text_count) VALUES (?, ?, 'completed', 2)",
        ("hash-clean", "/tmp/video.mp4"),
    )
    cursor.executemany(
        "INSERT INTO video_text (file_hash, raw_text, normalized_text, occurrence_count) VALUES (?, ?, ?, ?)",
        [
            ("hash-clean", "ok text", "ok text", 3),
            ("hash-clean", "bad", "bad", 2),
        ],
    )
    conn.commit()

    scanner_module.cleanup_ocr_text(4)

    rows = conn.execute("SELECT raw_text FROM video_text WHERE file_hash = 'hash-clean'").fetchall()
    assert [row[0] for row in rows] == ["ok text"]

    fragment_rows = conn.execute(
        "SELECT fragment_text FROM video_text_fragments WHERE file_hash = 'hash-clean'",
    ).fetchall()
    assert fragment_rows

    updated = conn.execute(
        "SELECT ocr_text_count FROM scanned_files WHERE file_hash = 'hash-clean'",
    ).fetchone()[0]
    assert updated == 1


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

    row = conn.execute(
        "SELECT person_name, suggested_person_name, suggested_confidence, suggestion_status, suggested_candidates FROM faces WHERE file_hash = 'h2'"
    ).fetchone()
    assert row[0] is None
    assert row[1] == "Alice"
    assert 0.6 <= row[2] <= 1
    assert row[3] == "pending"
    assert row[4] is not None


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

    # Use ThreadPoolExecutor instead of multiprocessing.Pool because
    # monkeypatch only affects the current process's namespace.
    # Threads share the same address space and will see the patched function.
    hashed_files = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        results_iterator = executor.map(scanner_module.get_file_hash_with_path, filepaths)
        for filepath, file_hash in results_iterator:
            hashed_files[filepath] = file_hash

    assert hashed_files == HASH_MAP


def test_classify_new_faces_skips_rejected(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    known = np.array([0.0, 0.0])
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name) VALUES (?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", pickle.dumps(known), "Alice"),
    )

    unknown = np.array([0.05, 0.0])
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, suggested_person_name, suggested_confidence, suggestion_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("h2", 0, "0,0,0,0", pickle.dumps(unknown), "Alice", 0.2, "rejected"),
    )
    conn.commit()

    monkeypatch.setattr(scanner_module.config, "AUTO_CLASSIFY_THRESHOLD", 0.3)
    scanner_module.classify_new_faces()

    row = conn.execute(
        "SELECT suggested_person_name, suggestion_status, suggested_candidates FROM faces WHERE file_hash = 'h2'"
    ).fetchone()
    assert row[0] == "Alice"
    assert row[1] == "rejected"
    assert row[2] is not None


def test_classify_new_faces_ambiguous_does_not_suggest(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    alice = np.array([0.0, 0.0])
    bob = np.array([0.03, 0.0])
    unknown = np.array([0.018, 0.0])

    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name) VALUES (?, ?, ?, ?, ?)",
        ("ha", 0, "0,0,0,0", pickle.dumps(alice), "Alice"),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name) VALUES (?, ?, ?, ?, ?)",
        ("hb", 0, "0,0,0,0", pickle.dumps(bob), "Bob"),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding) VALUES (?, ?, ?, ?)",
        ("hu", 0, "0,0,0,0", pickle.dumps(unknown)),
    )
    conn.commit()

    monkeypatch.setattr(scanner_module.config, "AUTO_CLASSIFY_THRESHOLD", 0.3)
    scanner_module.classify_new_faces()

    row = conn.execute(
        "SELECT suggested_person_name, suggested_confidence, suggestion_status, suggested_candidates FROM faces WHERE file_hash = 'hu'"
    ).fetchone()
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
    assert row[3] is not None


def test_unknown_faces_excluded_from_models(tmp_path, monkeypatch):
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)

    unknown_enc = np.array([0.0, 0.0])
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name, cluster_id) VALUES (?, ?, ?, ?, ?, ?)",
        ("hu", 0, "0,0,0,0", pickle.dumps(unknown_enc), "Unknown", -1),
    )

    unlabeled_enc = np.array([0.1, 0.1])
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        ("hx", 0, "0,0,0,0", pickle.dumps(unlabeled_enc), 5),
    )
    conn.commit()

    scanner_module.classify_new_faces()

    row = conn.execute(
        "SELECT suggested_person_name, suggested_candidates FROM faces WHERE file_hash = 'hx'"
    ).fetchone()
    assert row[0] is None
    assert row[1] is None
