import pickle
import sqlite3
from pathlib import Path

import numpy as np

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

    def run(self, stream, overwrite_output=True, quiet=True):  # pragma: no cover - used in writer tests
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
    assert plan.items[0].file_extension == '.mp4'
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
        (operation_id, "hash3", str(video_path), "pending", None, "People: Carol", "[\"Carol\"]", "[]"),
    )
    item_id = conn.execute("SELECT MAX(id) FROM metadata_operation_items").fetchone()[0]
    conn.commit()

    stub_ffmpeg = StubFFmpeg({str(video_path): "People: Carol"})
    backup_manager = metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg)
    record = backup_manager.create_backup(conn, "hash3", str(video_path), item_id)

    assert record.original_comment == "People: Carol"
    history_row = conn.execute("SELECT original_comment FROM metadata_history WHERE id = ?", (record.id,)).fetchone()
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
    operation_id = writer.start_operation([plan_item], metadata_services.WriteOptions(), background=False)

    history_service = metadata_services.HistoryService(str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg))
    data = history_service.get_operations()

    assert data['pagination']['total_items'] >= 1
    assert any(op['id'] == operation_id for op in data['operations'])

    details = history_service.get_operation_details(operation_id)
    assert details is not None
    assert details['operation']['id'] == operation_id
    assert details['items']


def test_history_service_rollback_restores_files(tmp_path, monkeypatch):
    db_path = _setup_test_database(tmp_path, monkeypatch)
    stub_ffmpeg = StubFFmpeg({})
    plan_item, video_path = _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg)

    writer = metadata_services.MetadataWriter(
        database_path=str(db_path),
        ffmpeg_module=stub_ffmpeg,
        backup_manager=metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg),
    )
    operation_id = writer.start_operation([plan_item], metadata_services.WriteOptions(), background=False)

    # Ensure comment was updated by writer stub
    assert stub_ffmpeg._comment_map.get(str(video_path)) == plan_item.result_comment

    with sqlite3.connect(db_path) as conn:
        backup_count = conn.execute("SELECT COUNT(*) FROM metadata_history").fetchone()[0]
    assert backup_count >= 1

    history_service = metadata_services.HistoryService(str(db_path), metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg))
    result = history_service.rollback_operation(operation_id)
    assert result['restored'] >= 1

    details = history_service.get_operation_details(operation_id)
    assert details['operation']['status'] == 'rolled_back'
    assert any(item['status'] == 'rolled_back' for item in details['items'])
    assert stub_ffmpeg._comment_map.get(str(video_path)) in {plan_item.existing_comment or '', None}
