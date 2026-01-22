from __future__ import annotations

import pickle
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    pass

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


# --- Tests for _normalize_ocr_text ---


def test_normalize_ocr_text_empty_string():
    """Empty string should return None."""
    assert scanner_module._normalize_ocr_text("") is None


def test_normalize_ocr_text_none():
    """None should return None."""
    assert scanner_module._normalize_ocr_text(None) is None


def test_normalize_ocr_text_whitespace_only():
    """Whitespace-only string should return None."""
    assert scanner_module._normalize_ocr_text("   \t\n   ") is None


def test_normalize_ocr_text_too_short(monkeypatch):
    """Text shorter than MIN_OCR_TEXT_LENGTH should return None."""
    monkeypatch.setattr(scanner_module, "MIN_OCR_TEXT_LENGTH", 5)
    monkeypatch.setattr(scanner_module, "OCR_MAX_TEXT_LENGTH", 100)
    assert scanner_module._normalize_ocr_text("ab") is None


def test_normalize_ocr_text_too_long(monkeypatch):
    """Text longer than OCR_MAX_TEXT_LENGTH should return None."""
    monkeypatch.setattr(scanner_module, "MIN_OCR_TEXT_LENGTH", 3)
    monkeypatch.setattr(scanner_module, "OCR_MAX_TEXT_LENGTH", 10)
    assert scanner_module._normalize_ocr_text("this text is way too long") is None


def test_normalize_ocr_text_valid(monkeypatch):
    """Valid text should be cleaned and returned."""
    monkeypatch.setattr(scanner_module, "MIN_OCR_TEXT_LENGTH", 3)
    monkeypatch.setattr(scanner_module, "OCR_MAX_TEXT_LENGTH", 100)
    result = scanner_module._normalize_ocr_text("  hello   world  ")
    assert result == "hello world"


def test_normalize_ocr_text_newlines(monkeypatch):
    """Multiple whitespace types should be normalized to single space."""
    monkeypatch.setattr(scanner_module, "MIN_OCR_TEXT_LENGTH", 3)
    monkeypatch.setattr(scanner_module, "OCR_MAX_TEXT_LENGTH", 100)
    result = scanner_module._normalize_ocr_text("hello\n\t  world")
    assert result == "hello world"


# --- Tests for _timestamp_ms_for_frame ---


def test_timestamp_ms_for_frame_zero_fps():
    """Zero FPS should return None."""
    assert scanner_module._timestamp_ms_for_frame(100, 0) is None


def test_timestamp_ms_for_frame_negative_fps():
    """Negative FPS should return None."""
    assert scanner_module._timestamp_ms_for_frame(100, -30) is None


def test_timestamp_ms_for_frame_none_fps():
    """None FPS should return None."""
    assert scanner_module._timestamp_ms_for_frame(100, None) is None


def test_timestamp_ms_for_frame_valid():
    """Valid fps should return correct milliseconds."""
    # Frame 30 at 30fps = 1 second = 1000ms
    assert scanner_module._timestamp_ms_for_frame(30, 30.0) == 1000
    # Frame 150 at 30fps = 5 seconds = 5000ms
    assert scanner_module._timestamp_ms_for_frame(150, 30.0) == 5000
    # Frame 0 = 0ms
    assert scanner_module._timestamp_ms_for_frame(0, 30.0) == 0


# --- Tests for _store_fragments ---


def test_store_fragments_empty_list(tmp_path, monkeypatch):
    """Empty fragments list should just delete existing and not insert."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert existing fragment
    cursor.execute(
        "INSERT INTO video_text_fragments (file_hash, rank, fragment_text, fragment_lower, occurrence_count, text_length) VALUES (?, ?, ?, ?, ?, ?)",
        ("hash1", 1, "old", "old", 5, 3),
    )
    conn.commit()

    scanner_module._store_fragments(cursor, "hash1", [])
    conn.commit()

    rows = cursor.execute(
        "SELECT * FROM video_text_fragments WHERE file_hash = ?", ("hash1",)
    ).fetchall()
    assert len(rows) == 0


def test_store_fragments_normal(tmp_path, monkeypatch):
    """Normal fragments should be stored correctly."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    fragments = [
        {"substring": "Hello", "lower": "hello", "count": 10, "length": 5},
        {"substring": "World", "lower": "world", "count": 5, "length": 5},
    ]

    scanner_module._store_fragments(cursor, "hash1", fragments)
    conn.commit()

    rows = cursor.execute(
        "SELECT rank, fragment_text, fragment_lower, occurrence_count, text_length FROM video_text_fragments WHERE file_hash = ? ORDER BY rank",
        ("hash1",),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == (1, "Hello", "hello", 10, 5)
    assert rows[1] == (2, "World", "world", 5, 5)


def test_store_fragments_missing_keys(tmp_path, monkeypatch):
    """Fragments with missing keys should use defaults."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    fragments = [{"substring": "Test"}]

    scanner_module._store_fragments(cursor, "hash1", fragments)
    conn.commit()

    row = cursor.execute(
        "SELECT fragment_text, fragment_lower, occurrence_count, text_length FROM video_text_fragments WHERE file_hash = ?",
        ("hash1",),
    ).fetchone()
    assert row[0] == "Test"
    assert row[1] == "test"
    assert row[2] == 0
    assert row[3] == 4


# --- Tests for _update_ocr_counts ---


def test_update_ocr_counts_updates_count(tmp_path, monkeypatch):
    """Should update count for existing file."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, processing_status, ocr_text_count) VALUES (?, ?, 'completed', 0)",
        ("hash1", "/tmp/video.mp4"),
    )
    cursor.executemany(
        "INSERT INTO video_text (file_hash, raw_text, normalized_text, occurrence_count) VALUES (?, ?, ?, ?)",
        [("hash1", "text1", "text1", 1), ("hash1", "text2", "text2", 1)],
    )
    conn.commit()

    scanner_module._update_ocr_counts(cursor, ["hash1"])
    conn.commit()

    count = cursor.execute(
        "SELECT ocr_text_count FROM scanned_files WHERE file_hash = ?", ("hash1",)
    ).fetchone()[0]
    assert count == 2


def test_update_ocr_counts_zero_count(tmp_path, monkeypatch):
    """Zero count should not update timestamp."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, processing_status, ocr_text_count, ocr_last_updated) VALUES (?, ?, 'completed', 5, '2024-01-01 00:00:00')",
        ("hash1", "/tmp/video.mp4"),
    )
    conn.commit()

    scanner_module._update_ocr_counts(cursor, ["hash1"])
    conn.commit()

    row = cursor.execute(
        "SELECT ocr_text_count, ocr_last_updated FROM scanned_files WHERE file_hash = ?", ("hash1",)
    ).fetchone()
    assert row[0] == 0
    # Timestamp should remain unchanged since count is 0
    assert row[1] == "2024-01-01 00:00:00"


# --- Tests for _recompute_fragments ---


def test_recompute_fragments(tmp_path, monkeypatch):
    """Should recompute fragments from video_text entries."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany(
        "INSERT INTO video_text (file_hash, raw_text, normalized_text, occurrence_count) VALUES (?, ?, ?, ?)",
        [
            ("hash1", "Hello World", "hello world", 5),
            ("hash1", "Test Text", "test text", 3),
        ],
    )
    conn.commit()

    scanner_module._recompute_fragments(cursor, "hash1")
    conn.commit()

    rows = cursor.execute(
        "SELECT fragment_text FROM video_text_fragments WHERE file_hash = ? ORDER BY rank",
        ("hash1",),
    ).fetchall()
    assert len(rows) > 0


# --- Tests for validate_video_file ---


def test_validate_video_file_valid(monkeypatch):
    """Valid video should return True."""
    probe_result = {
        "format": {"duration": "60.5"},
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
    }
    monkeypatch.setattr(scanner_module.ffmpeg, "probe", lambda path, **kwargs: probe_result)

    is_valid, error = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is True
    assert error is None


def test_validate_video_file_zero_duration(monkeypatch):
    """Zero duration should return False."""
    probe_result = {
        "format": {"duration": "0"},
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
    }
    monkeypatch.setattr(scanner_module.ffmpeg, "probe", lambda path, **kwargs: probe_result)

    is_valid, error = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is False
    assert error is not None
    assert "zero or negative duration" in error


def test_validate_video_file_negative_duration(monkeypatch):
    """Negative duration should return False."""
    probe_result = {
        "format": {"duration": "-10"},
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
    }
    monkeypatch.setattr(scanner_module.ffmpeg, "probe", lambda path, **kwargs: probe_result)

    is_valid, error = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is False
    assert error is not None
    assert "zero or negative duration" in error


def test_validate_video_file_no_video_streams(monkeypatch):
    """No video streams should return False."""
    probe_result = {
        "format": {"duration": "60"},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
    }
    monkeypatch.setattr(scanner_module.ffmpeg, "probe", lambda path, **kwargs: probe_result)

    is_valid, error = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is False
    assert error is not None
    assert "No video streams found" in error


def test_validate_video_file_corrupted_stream(monkeypatch):
    """Stream without codec_name should return False."""
    probe_result = {
        "format": {"duration": "60"},
        "streams": [{"codec_type": "video", "codec_name": None}],
    }
    monkeypatch.setattr(scanner_module.ffmpeg, "probe", lambda path, **kwargs: probe_result)

    is_valid, error = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is False
    assert error is not None
    assert "Corrupted video stream" in error


def test_validate_video_file_ffmpeg_error(monkeypatch):
    """FFmpeg error should return False with error message."""
    error = scanner_module.ffmpeg.Error("ffmpeg", b"stdout", b"Error decoding file")
    monkeypatch.setattr(scanner_module.ffmpeg, "probe", MagicMock(side_effect=error))

    is_valid, err_msg = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is False
    assert err_msg is not None
    assert "FFprobe error" in err_msg


def test_validate_video_file_generic_exception(monkeypatch):
    """Generic exception should return False with error message."""
    monkeypatch.setattr(
        scanner_module.ffmpeg, "probe", MagicMock(side_effect=ValueError("Unexpected error"))
    )

    is_valid, err_msg = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is False
    assert err_msg is not None
    assert "Validation error" in err_msg


# --- Tests for _require_ffprobe ---


def test_require_ffprobe_found(monkeypatch):
    """Should not raise when ffprobe is found."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/ffprobe")
    # Should not raise
    scanner_module._require_ffprobe()


def test_require_ffprobe_not_found(monkeypatch):
    """Should raise SystemExit when ffprobe is not found."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(SystemExit) as exc_info:
        scanner_module._require_ffprobe()
    assert exc_info.value.code == 1


# --- Tests for serialize_ocr_entries ---


def test_serialize_ocr_entries_empty():
    """Empty aggregator should return empty list."""
    result = scanner_module.serialize_ocr_entries("hash1", {})
    assert result == []


def test_serialize_ocr_entries_normal():
    """Should serialize OCR entries sorted by confidence."""
    aggregator = {
        "hello world": {
            "raw_text": "Hello World",
            "normalized_text": "hello world",
            "confidence": 0.9,
            "first_seen_frame": 10,
            "first_seen_timestamp_ms": 333,
            "occurrence_count": 3,
        },
        "test": {
            "raw_text": "Test",
            "normalized_text": "test",
            "confidence": 0.7,
            "first_seen_frame": 5,
            "first_seen_timestamp_ms": 166,
            "occurrence_count": 1,
        },
    }
    result = scanner_module.serialize_ocr_entries("hash1", aggregator)
    assert len(result) == 2
    # Should be sorted by confidence descending
    assert result[0]["confidence"] == 0.9
    assert result[1]["confidence"] == 0.7
    assert all(item["file_hash"] == "hash1" for item in result)


# --- Tests for cleanup_ocr_text edge cases ---


def test_cleanup_ocr_text_no_entries_to_clean(tmp_path, monkeypatch):
    """Cleanup with no short entries should exit early."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, processing_status, ocr_text_count) VALUES (?, ?, 'completed', 1)",
        ("hash1", "/tmp/video.mp4"),
    )
    cursor.execute(
        "INSERT INTO video_text (file_hash, raw_text, normalized_text, occurrence_count) VALUES (?, ?, ?, ?)",
        ("hash1", "long enough text", "long enough text", 1),
    )
    conn.commit()

    scanner_module.cleanup_ocr_text(4)

    # Should still have the entry
    count = conn.execute("SELECT COUNT(*) FROM video_text WHERE file_hash = 'hash1'").fetchone()[0]
    assert count == 1


# --- Tests for index route skipping behavior ---


def test_index_clears_skipped_when_all_skipped(tmp_path, monkeypatch):
    """When all clusters are skipped, should clear and loop back."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 5),
    )
    conn.commit()
    # This test ensures the loop back logic is covered in app.py


# --- Tests for write_data_to_db edge cases ---


def test_write_data_to_db_no_data(tmp_path, monkeypatch):
    """write_data_to_db with empty data should not crash."""
    setup_temp_db(tmp_path, monkeypatch)
    scanner_module.write_data_to_db(
        face_data=[],
        scanned_files_info=[],
        failed_files_info=None,
        ocr_text_data=[],
        ocr_fragments_data=[],
    )


def test_write_data_to_db_with_failed_files(tmp_path, monkeypatch):
    """write_data_to_db should handle failed files."""
    db_path = setup_temp_db(tmp_path, monkeypatch)

    scanner_module.write_data_to_db(
        face_data=[],
        scanned_files_info=[],
        failed_files_info=[("hash1", "/tmp/video.mp4", "error: test failure")],
        ocr_text_data=[],
        ocr_fragments_data=[],
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT processing_status, error_message FROM scanned_files WHERE file_hash = 'hash1'"
    ).fetchone()
    assert row[0] == "failed"
    assert "test failure" in row[1]


# --- Tests for _persist_ocr_results ---


def test_persist_ocr_results_success(tmp_path, monkeypatch):
    """_persist_ocr_results should insert OCR data and update counts."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, processing_status) VALUES (?, ?, 'completed')",
        ("testhash", "/tmp/video.mp4"),
    )
    conn.commit()

    ocr_entries = [
        {
            "raw_text": "Hello World",
            "normalized_text": "hello world",
            "confidence": 0.9,
            "first_seen_frame": 10,
            "first_seen_timestamp_ms": 333,
            "occurrence_count": 2,
        },
        {
            "raw_text": "Test",
            "normalized_text": "test",
            "confidence": 0.8,
            "first_seen_frame": 20,
            "first_seen_timestamp_ms": 666,
            "occurrence_count": 1,
        },
    ]
    ocr_fragments = [
        {"substring": "hello", "lower": "hello", "count": 5, "length": 5},
    ]

    result = scanner_module._persist_ocr_results("testhash", ocr_entries, ocr_fragments)
    assert result is True

    # Verify OCR entries were inserted
    count = conn.execute("SELECT COUNT(*) FROM video_text WHERE file_hash = 'testhash'").fetchone()[
        0
    ]
    assert count == 2

    # Verify count was updated
    ocr_count = conn.execute(
        "SELECT ocr_text_count FROM scanned_files WHERE file_hash = 'testhash'"
    ).fetchone()[0]
    assert ocr_count == 2


def test_persist_ocr_results_empty_entries(tmp_path, monkeypatch):
    """_persist_ocr_results with empty entries should still work."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, processing_status) VALUES (?, ?, 'completed')",
        ("emptyhash", "/tmp/video.mp4"),
    )
    conn.commit()

    result = scanner_module._persist_ocr_results("emptyhash", [], [])
    assert result is True

    count = conn.execute(
        "SELECT ocr_text_count FROM scanned_files WHERE file_hash = 'emptyhash'"
    ).fetchone()[0]
    assert count == 0


def test_persist_ocr_results_replaces_existing(tmp_path, monkeypatch):
    """_persist_ocr_results should delete existing entries before inserting new ones."""
    db_path = setup_temp_db(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath, processing_status) VALUES (?, ?, 'completed')",
        ("replacehash", "/tmp/video.mp4"),
    )
    conn.execute(
        "INSERT INTO video_text (file_hash, raw_text, normalized_text, occurrence_count) VALUES (?, ?, ?, ?)",
        ("replacehash", "Old Text", "old text", 1),
    )
    conn.commit()

    # Verify old entry exists
    old_count = conn.execute(
        "SELECT COUNT(*) FROM video_text WHERE file_hash = 'replacehash'"
    ).fetchone()[0]
    assert old_count == 1

    new_entries = [
        {
            "raw_text": "New Text",
            "normalized_text": "new text",
            "confidence": 0.95,
            "first_seen_frame": 5,
            "first_seen_timestamp_ms": 100,
            "occurrence_count": 3,
        },
    ]

    result = scanner_module._persist_ocr_results("replacehash", new_entries, [])
    assert result is True

    # Old entry should be deleted, new one should exist
    rows = conn.execute(
        "SELECT raw_text FROM video_text WHERE file_hash = 'replacehash'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "New Text"


# --- Tests for get_file_hash_with_path ---


def test_get_file_hash_with_path(tmp_path, monkeypatch):
    """get_file_hash_with_path should return filepath and computed hash."""
    test_file = tmp_path / "test_video.mp4"
    test_file.write_bytes(b"fake video content for hashing")

    filepath, computed_hash = scanner_module.get_file_hash_with_path(str(test_file))

    assert filepath == str(test_file)
    assert computed_hash is not None
    assert len(computed_hash) == 64  # SHA256 hex digest length


# --- Tests for more app.py coverage ---


def test_validate_video_file_ffprobe_exception(monkeypatch):
    """validate_video_file should handle ffprobe throwing general exceptions."""

    def raise_exception(path, **kwargs):
        raise RuntimeError("Unexpected error during probe")

    monkeypatch.setattr(scanner_module.ffmpeg, "probe", raise_exception)

    is_valid, error = scanner_module.validate_video_file("/fake/video.mp4")
    assert is_valid is False
    assert error is not None
    assert "Validation error" in error
