import os
import shutil
import sqlite3
from pathlib import Path

import pytest


def _collect_file_stats(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def _video_files(root: Path) -> list[Path]:
    extensions = {".mp4", ".mkv", ".mov", ".avi"}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]


def test_e2e_pipeline_runs_on_copied_videos(tmp_path, monkeypatch):
    """Run the full pipeline on a copied dataset and verify output artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = repo_root / "test_vids"
    if not input_dir.exists():
        pytest.skip("test_vids dataset not available")

    pytest.importorskip("ffmpeg")
    try:
        import face_recognition  # noqa: F401
    except Exception:
        pytest.skip("face_recognition unavailable for end-to-end scan")

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not available on PATH")

    input_files = [path for path in input_dir.rglob("*") if path.is_file()]
    input_stats_before = _collect_file_stats(input_files)

    work_dir = tmp_path / "e2e_work"
    video_dir = work_dir / "videos"
    shutil.copytree(input_dir, video_dir)

    monkeypatch.setenv("INDEXIUM_VIDEO_DIR", str(video_dir))
    monkeypatch.setenv("INDEXIUM_DB", str(work_dir / "faces.db"))
    monkeypatch.setenv("CPU_CORES", "1")
    monkeypatch.setenv("METADATA_PLAN_WORKERS", "1")
    monkeypatch.setenv("DBSCAN_MIN_SAMPLES", "1")

    import e2e_test

    e2e_test.run_pipeline(str(video_dir), str(work_dir))

    input_stats_after = _collect_file_stats(input_files)
    assert input_stats_after == input_stats_before

    db_path = work_dir / "faces.db"
    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        scanned_files = conn.execute("SELECT last_known_filepath FROM scanned_files").fetchall()
        face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        cluster_count = conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]

        assert face_count >= 0
        if _video_files(video_dir):
            assert len(scanned_files) >= 1

        if cluster_count > 0:
            named_faces = conn.execute(
                "SELECT COUNT(*) FROM faces WHERE person_name = ?",
                ("Test Person",),
            ).fetchone()[0]
            assert named_faces > 0

            operation = conn.execute(
                "SELECT status, success_count, failure_count FROM metadata_operations ORDER BY id DESC LIMIT 1"
            ).fetchone()
            assert operation is not None
            status, success_count, failure_count = operation
            assert status in {"completed", "in_progress", "cancelled"}
            assert success_count >= 0
            assert failure_count >= 0
