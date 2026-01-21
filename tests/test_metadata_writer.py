import pickle
import sqlite3
from pathlib import Path

import numpy as np

import metadata_services
import scanner as scanner_module


class StubFFmpeg:
    """Minimal ffmpeg stub to simulate comment reads/writes."""

    class Error(Exception):
        pass

    def __init__(self, comments=None):
        self._comments = comments or {}
        self.outputs = []

    def probe(self, path):
        return {"format": {"tags": {"comment": self._comments.get(path, "")}}}

    def input(self, path):  # pragma: no cover - simple holder
        return {"input_path": path}

    def output(self, stream, output_path, **kwargs):  # pragma: no cover - simple holder
        return {
            "input_path": stream["input_path"],
            "output_path": output_path,
            "kwargs": kwargs,
        }

    def run(self, stream, overwrite_output=True, quiet=True):
        temp = Path(stream["output_path"])
        temp.write_text("stub")
        self.outputs.append(stream)
        metadata = stream["kwargs"].get("metadata")
        if isinstance(metadata, str) and metadata.startswith("comment="):
            comment_value = metadata.split("=", 1)[1]
            self._comments[stream["input_path"]] = comment_value


def _setup_database(tmp_path, monkeypatch):
    db_path = tmp_path / "writer.db"
    monkeypatch.setattr(scanner_module.config, "DATABASE_FILE", str(db_path))
    monkeypatch.setattr(scanner_module, "DATABASE_FILE", str(db_path))
    scanner_module.setup_database()
    return db_path


def _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg):
    conn = sqlite3.connect(db_path)
    enc = pickle.dumps(np.array([0]))
    video_path = tmp_path / "writer_video.mp4"
    video_path.write_text("video")

    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hashw", str(video_path)),
    )
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, cluster_id, person_name) VALUES (?, ?, ?, ?, ?, ?)",
        ("hashw", 0, "0,0,0,0", enc, 1, "Writer"),
    )
    conn.commit()
    conn.close()

    planner = metadata_services.MetadataPlanner(ffmpeg_module=stub_ffmpeg)
    with sqlite3.connect(db_path) as conn:
        plan = planner.generate_plan(conn)
    return plan.items[0], video_path


def test_metadata_writer_completes_operation(tmp_path, monkeypatch):
    db_path = _setup_database(tmp_path, monkeypatch)
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
    status = writer.get_operation_status(operation_id)

    assert status is not None
    assert status["status"] == "completed"
    assert status["success_count"] == 1
    assert status["failure_count"] == 0
    assert Path(video_path).exists()

    with sqlite3.connect(db_path) as conn:
        history_rows = conn.execute("SELECT COUNT(*) FROM metadata_history").fetchone()
    assert history_rows[0] == 1


def test_metadata_writer_handles_missing_file(tmp_path, monkeypatch):
    db_path = _setup_database(tmp_path, monkeypatch)
    stub_ffmpeg = StubFFmpeg({})
    plan_item, video_path = _create_plan_item(tmp_path, db_path, monkeypatch, stub_ffmpeg)
    video_path.unlink()  # simulate missing file at write time

    writer = metadata_services.MetadataWriter(
        database_path=str(db_path),
        ffmpeg_module=stub_ffmpeg,
        backup_manager=metadata_services.BackupManager(ffmpeg_module=stub_ffmpeg),
    )

    operation_id = writer.start_operation(
        [plan_item], metadata_services.WriteOptions(), background=False
    )
    status = writer.get_operation_status(operation_id)

    assert status is not None
    assert status["status"] == "completed"
    assert status["failure_count"] == 1
    assert status["items"][0]["status"] == "failed"
