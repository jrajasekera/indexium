from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

import metadata_services
import scanner as scanner_module


class StubFFmpeg:
    """Lightweight ffmpeg stub for planner and backup tests."""

    class Error(Exception):
        pass

    def __init__(self, comment_map=None):
        self._comment_map = comment_map or {}

    def probe(self, path):
        comment = self._comment_map.get(path, "")
        return {"format": {"tags": {"comment": comment}}}

    # The planner does not call input/output/run; these exist for completeness.
    def input(self, path):  # pragma: no cover - defensive stub
        return {"input_path": path}

    def output(self, stream, output_path, **kwargs):  # pragma: no cover - defensive stub
        return {"input_path": stream["input_path"], "output_path": output_path, "kwargs": kwargs}

    def run(
        self, stream, overwrite_output=True, quiet=True
    ):  # pragma: no cover - used in writer tests
        temp = Path(stream["output_path"])
        temp.write_text("stub")
        metadata = stream["kwargs"].get("metadata")
        if isinstance(metadata, str) and metadata.startswith("comment="):
            self._comment_map[stream["input_path"]] = metadata.split("=", 1)[1]


def _setup_test_database(tmp_path, monkeypatch):
    db_path = tmp_path / "planner.db"
    monkeypatch.setattr(scanner_module.config, "DATABASE_FILE", str(db_path))
    monkeypatch.setattr(scanner_module, "DATABASE_FILE", str(db_path))
    scanner_module.setup_database()
    return db_path


def test_metadata_planner_generates_plan_with_risk_counts(tmp_path, monkeypatch):
    db_path = _setup_test_database(tmp_path, monkeypatch)
    existing_video = tmp_path / "video_exists.mp4"
    existing_video.write_text("data")

    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash1", str(existing_video)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("hash1", 0, "0,0,0,0", enc, 1, "Alice"),
    )

    # Missing file entry should be flagged as blocked/danger
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash2", str(tmp_path / "missing.mp4")),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("hash2", 0, "0,0,0,0", enc, 2, "Bob"),
    )
    conn.commit()

    stub_ffmpeg = StubFFmpeg({str(existing_video): "People: Alice"})
    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub_ffmpeg)
    plan = planner.generate_plan(conn)

    assert plan.statistics.total_files == 2
    assert plan.statistics.blocked_count == 1
    assert plan.statistics.safe_count == 1
    assert sorted(item.file_hash for item in plan.categories.get("danger", [])) == ["hash2"]
    assert plan.items[0].result_people == ["Alice"]
    assert plan.items[0].file_extension == ".mp4"
    assert plan.items[0].tag_count == 1

    filtered = planner.filter_items(plan.items, {"risk_levels": ["safe"]})
    assert len(filtered) == 1
    assert filtered[0].file_hash == "hash1"

    file_type_filtered = planner.filter_items(plan.items, {"file_types": ["mp4"]})
    assert {item.file_hash for item in file_type_filtered} == {"hash1", "hash2"}

    issue_filtered = planner.filter_items(plan.items, {"issue_codes": ["missing_file"]})
    assert len(issue_filtered) == 1
    assert issue_filtered[0].file_hash == "hash2"

    tag_filtered = planner.filter_items(plan.items, {"tag_count": {"min": 1, "max": 1}})
    assert {item.file_hash for item in tag_filtered} == {"hash1", "hash2"}

    sorted_by_risk = planner.sort_items(plan.items, sort_by="risk", direction="desc")
    assert sorted_by_risk[0].file_hash == "hash2"

    conn.close()


def test_backup_manager_create_backup_records_original_comment(tmp_path, monkeypatch):
    db_path = _setup_test_database(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video_path = tmp_path / "restore_me.mp4"
    video_path.write_text("video")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash3", str(video_path)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("hash3", 0, "0,0,0,0", enc, 3, "Carol"),
    )
    conn.execute(
        "INSERT INTO metadata_operations (operation_type, status) VALUES (?, ?)",
        ("write", "pending"),
    )
    operation_id = conn.execute("SELECT MAX(id) FROM metadata_operations").fetchone()[0]
    conn.execute(
        """
        INSERT INTO metadata_operation_items (operation_id, file_hash, file_path, status, previous_comment, new_comment, tags_added, tags_removed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            operation_id,
            "hash3",
            str(video_path),
            "pending",
            None,
            "People: Carol",
            '["Carol"]',
            "[]",
        ),
    )
    item_id = conn.execute("SELECT MAX(id) FROM metadata_operation_items").fetchone()[0]
    conn.commit()

    stub_ffmpeg = StubFFmpeg({str(video_path): "People: Carol"})
    backup_manager = metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    record = backup_manager.create_backup(conn, "hash3", str(video_path), item_id)

    assert record.original_comment == "People: Carol"
    history_row = conn.execute(
        "SELECT original_comment FROM metadata_history WHERE id = ?", (record.id,)
    ).fetchone()
    assert history_row[0] == "People: Carol"

    conn.close()


def _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg):
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    video_path = tmp_path / "history_video.mp4"
    video_path.write_text("video")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hist1", str(video_path)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("hist1", 0, "0,0,0,0", enc, 5, "History"),
    )
    conn.commit()
    conn.close()

    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub_ffmpeg)
    with sqlite3.connect(db_path) as conn:
        plan = planner.generate_plan(conn)
    return plan.items[0], video_path


def test_history_service_lists_operations(tmp_path, monkeypatch):
    db_path = _setup_test_database(tmp_path, monkeypatch)
    stub_ffmpeg = StubFFmpeg({})
    plan_item, _ = _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg)

    writer = metadata_services.MetadataWriter(
        database_path=str(db_path),
        ffmpeg_module=stub_ffmpeg,
        backup_manager=metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg),
    )
    operation_id = writer.start_operation(
        [plan_item], metadata_services.WriteOptions(), background=False
    )

    history_service = metadata_services.HistoryService(
        str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    )
    data = history_service.get_operations()

    assert data["pagination"]["total_items"] >= 1
    assert any(op["id"] == operation_id for op in data["operations"])

    details = history_service.get_operation_details(operation_id)
    assert details is not None
    assert details["operation"]["id"] == operation_id
    assert details["items"]


def test_history_service_rollback_restores_files(tmp_path, monkeypatch):
    db_path = _setup_test_database(tmp_path, monkeypatch)
    stub_ffmpeg = StubFFmpeg({})
    plan_item, video_path = _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg)

    writer = metadata_services.MetadataWriter(
        database_path=str(db_path),
        ffmpeg_module=stub_ffmpeg,
        backup_manager=metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg),
    )
    operation_id = writer.start_operation(
        [plan_item], metadata_services.WriteOptions(), background=False
    )

    # Ensure comment was updated by writer stub
    assert stub_ffmpeg._comment_map.get(str(video_path)) == plan_item.result_comment

    with sqlite3.connect(db_path) as conn:
        backup_count = conn.execute("SELECT COUNT(*) FROM metadata_history").fetchone()[0]
    assert backup_count >= 1

    history_service = metadata_services.HistoryService(
        str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    )
    result = history_service.rollback_operation(operation_id)
    assert result["restored"] >= 1

    details = history_service.get_operation_details(operation_id)
    assert details is not None
    assert details["operation"]["status"] == "rolled_back"
    assert any(item["status"] == "rolled_back" for item in details["items"])
    assert stub_ffmpeg._comment_map.get(str(video_path)) in {plan_item.existing_comment or "", None}


# --- Additional tests for filter_items ---


def test_filter_items_requires_update(tmp_path, monkeypatch):
    """Should filter by requires_update field."""
    db_path = _setup_test_database(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video1 = tmp_path / "video1.mp4"
    video1.write_text("v1")
    video2 = tmp_path / "video2.mp4"
    video2.write_text("v2")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h1", str(video1)),
    )
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h2", str(video2)),
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

    # video1 has matching comment, video2 doesn't
    stub = StubFFmpeg({str(video1): "People: Alice", str(video2): ""})
    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub)
    plan = planner.generate_plan(conn)

    # Filter for items that require update
    filtered = planner.filter_items(plan.items, {"requires_update": True})
    assert len(filtered) == 1
    assert filtered[0].file_hash == "h2"

    # Filter for items that don't require update
    no_update = planner.filter_items(plan.items, {"requires_update": False})
    assert len(no_update) == 1
    assert no_update[0].file_hash == "h1"


def test_filter_items_can_update(tmp_path, monkeypatch):
    """Should filter by can_update field."""
    db_path = _setup_test_database(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video1 = tmp_path / "video1.mp4"
    video1.write_text("v1")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h1", str(video1)),
    )
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h2", str(tmp_path / "missing.mp4")),  # Missing file
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

    stub = StubFFmpeg({str(video1): ""})
    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub)
    plan = planner.generate_plan(conn)

    # Filter for items that can be updated
    can_update = planner.filter_items(plan.items, {"can_update": True})
    assert len(can_update) == 1
    assert can_update[0].file_hash == "h1"

    # Filter for items that cannot be updated
    cannot_update = planner.filter_items(plan.items, {"can_update": False})
    assert len(cannot_update) == 1
    assert cannot_update[0].file_hash == "h2"


def test_filter_items_search(tmp_path, monkeypatch):
    """Should filter by search string matching file name, people, or hash."""
    db_path = _setup_test_database(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video1 = tmp_path / "conference_recording.mp4"
    video1.write_text("v1")
    video2 = tmp_path / "birthday_party.mp4"
    video2.write_text("v2")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("confhash", str(video1)),
    )
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("birthhash", str(video2)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("confhash", 0, "0,0,0,0", enc, 1, "Alice Smith"),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("birthhash", 0, "0,0,0,0", enc, 2, "Bob Jones"),
    )
    conn.commit()

    stub = StubFFmpeg({str(video1): "", str(video2): ""})
    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub)
    plan = planner.generate_plan(conn)

    # Search by file name
    by_name = planner.filter_items(plan.items, {"search": "conference"})
    assert len(by_name) == 1
    assert by_name[0].file_hash == "confhash"

    # Search by person name
    by_person = planner.filter_items(plan.items, {"search": "alice"})
    assert len(by_person) == 1
    assert by_person[0].file_hash == "confhash"

    # Search by hash
    by_hash = planner.filter_items(plan.items, {"search": "birth"})
    assert len(by_hash) == 1
    assert by_hash[0].file_hash == "birthhash"


# --- Additional tests for sort_items ---


def test_sort_items_by_tag_count(tmp_path, monkeypatch):
    """Should sort by tag count."""
    db_path = _setup_test_database(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video1 = tmp_path / "video1.mp4"
    video1.write_text("v1")
    video2 = tmp_path / "video2.mp4"
    video2.write_text("v2")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h1", str(video1)),
    )
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h2", str(video2)),
    )
    # h1 has 1 person
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("h1", 0, "0,0,0,0", enc, 1, "Alice"),
    )
    # h2 has 2 people
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("h2", 0, "0,0,0,0", enc, 2, "Bob"),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("h2", 1, "0,0,0,0", enc, 3, "Carol"),
    )
    conn.commit()

    stub = StubFFmpeg({str(video1): "", str(video2): ""})
    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub)
    plan = planner.generate_plan(conn)

    # Sort ascending
    sorted_asc = planner.sort_items(plan.items, sort_by="tag_count", direction="asc")
    assert sorted_asc[0].tag_count <= sorted_asc[1].tag_count

    # Sort descending
    sorted_desc = planner.sort_items(plan.items, sort_by="tag_count", direction="desc")
    assert sorted_desc[0].tag_count >= sorted_desc[1].tag_count


def test_sort_items_by_modified(tmp_path, monkeypatch):
    """Should sort by file modification time."""
    import time

    db_path = _setup_test_database(tmp_path, monkeypatch)
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))

    video1 = tmp_path / "video1.mp4"
    video1.write_text("v1")
    time.sleep(0.1)  # Ensure different mtime
    video2 = tmp_path / "video2.mp4"
    video2.write_text("v2")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h1", str(video1)),
    )
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("h2", str(video2)),
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

    stub = StubFFmpeg({str(video1): "", str(video2): ""})
    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub)
    plan = planner.generate_plan(conn)

    # Sort ascending (oldest first)
    sorted_asc = planner.sort_items(plan.items, sort_by="modified", direction="asc")
    assert sorted_asc[0].file_hash == "h1"

    # Sort descending (newest first)
    sorted_desc = planner.sort_items(plan.items, sort_by="modified", direction="desc")
    assert sorted_desc[0].file_hash == "h2"


# --- Tests for HistoryService with filters ---


def test_get_operations_with_status_filter(tmp_path, monkeypatch):
    """Should filter operations by status."""
    db_path = _setup_test_database(tmp_path, monkeypatch)
    stub_ffmpeg = StubFFmpeg({})

    # Create an operation
    plan_item, _ = _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg)
    writer = metadata_services.MetadataWriter(
        database_path=str(db_path),
        ffmpeg_module=stub_ffmpeg,
        backup_manager=metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg),
    )
    operation_id = writer.start_operation(
        [plan_item], metadata_services.WriteOptions(), background=False
    )

    history_service = metadata_services.HistoryService(
        str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    )

    # Filter by completed status
    completed = history_service.get_operations(filters={"status": ["completed"]})
    assert any(op["id"] == operation_id for op in completed["operations"])

    # Filter by pending status (should not include our completed operation)
    pending = history_service.get_operations(filters={"status": ["pending"]})
    assert not any(op["id"] == operation_id for op in pending["operations"])


def test_get_operations_with_date_filter(tmp_path, monkeypatch):
    """Should filter operations by date range."""
    db_path = _setup_test_database(tmp_path, monkeypatch)
    stub_ffmpeg = StubFFmpeg({})

    plan_item, _ = _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg)
    writer = metadata_services.MetadataWriter(
        database_path=str(db_path),
        ffmpeg_module=stub_ffmpeg,
        backup_manager=metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg),
    )
    writer.start_operation([plan_item], metadata_services.WriteOptions(), background=False)

    history_service = metadata_services.HistoryService(
        str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    )

    from datetime import date, timedelta

    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Filter starting from today should include our operation
    today_ops = history_service.get_operations(filters={"start_date": today})
    assert today_ops["pagination"]["total_items"] >= 1

    # Filter ending yesterday should not include our operation
    old_ops = history_service.get_operations(filters={"end_date": yesterday})
    # The test creates a new operation today, so it shouldn't be in results ending yesterday
    assert old_ops["pagination"]["total_items"] == 0


def test_rollback_invalid_operation(tmp_path, monkeypatch):
    """Should raise error when rollback on non-existent operation."""
    db_path = _setup_test_database(tmp_path, monkeypatch)
    stub_ffmpeg = StubFFmpeg({})

    history_service = metadata_services.HistoryService(
        str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    )

    import pytest

    with pytest.raises(ValueError, match="Operation not found"):
        history_service.rollback_operation(99999)


def test_rollback_running_operation(tmp_path, monkeypatch):
    """Should raise error when trying to rollback a running operation."""
    db_path = _setup_test_database(tmp_path, monkeypatch)

    # Manually create a pending operation
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO metadata_operations (operation_type, status) VALUES (?, ?)",
        ("write", "pending"),
    )
    operation_id = conn.execute("SELECT MAX(id) FROM metadata_operations").fetchone()[0]
    conn.commit()
    conn.close()

    stub_ffmpeg = StubFFmpeg({})
    history_service = metadata_services.HistoryService(
        str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    )

    import pytest

    with pytest.raises(ValueError, match="still running"):
        history_service.rollback_operation(operation_id)
