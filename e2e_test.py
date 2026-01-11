import argparse
import os
import shutil
import sqlite3
import tempfile
import time
from urllib.parse import urlparse, parse_qs

import ffmpeg


def run_pipeline(video_dir: str, work_dir: str) -> None:
    """Run the full pipeline on videos in *video_dir* using *work_dir*."""
    db_path = os.path.join(work_dir, "faces.db")
    thumb_dir = os.path.join(work_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    os.environ["INDEXIUM_VIDEO_DIR"] = video_dir
    os.environ["INDEXIUM_DB"] = db_path
    os.environ.setdefault("DBSCAN_MIN_SAMPLES", "1")
    os.environ.setdefault("CPU_CORES", "1")

    from signal_handler import SignalHandler
    import scanner
    import app as app_module

    # ensure modules use our temporary paths
    scanner.config.THUMBNAIL_DIR = thumb_dir
    scanner.config.DATABASE_FILE = db_path
    scanner.DATABASE_FILE = db_path
    app_module.config.THUMBNAIL_DIR = thumb_dir
    app_module.config.DATABASE_FILE = db_path

    scanner.setup_database()
    handler = SignalHandler()
    scanner.scan_videos_parallel(handler)
    scanner.classify_new_faces()
    scanner.cluster_faces()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL ORDER BY cluster_id LIMIT 1"
        ).fetchone()

    if row:
        cluster_id = row[0]
        with app_module.app.test_client() as client:
            client.post(
                "/name_cluster",
                data={"cluster_id": cluster_id, "person_name": "Test Person"},
                follow_redirects=True,
            )
            file_hashes = []
            with sqlite3.connect(db_path) as conn:
                file_hashes = [
                    row[0]
                    for row in conn.execute(
                        "SELECT DISTINCT file_hash FROM faces WHERE person_name IS NOT NULL"
                    ).fetchall()
                ]
            if file_hashes:
                response = client.post(
                    "/write_metadata",
                    data={"file_hashes": file_hashes},
                    follow_redirects=False,
                )
                location = response.headers.get("Location", "")
                query = parse_qs(urlparse(location).query)
                operation_ids = query.get("operation_id", [])
                if operation_ids:
                    operation_id = int(operation_ids[0])
                    timeout = time.monotonic() + 60
                    while time.monotonic() < timeout:
                        status = app_module.metadata_writer.get_operation_status(operation_id)
                        if not status:
                            break
                        if status["status"] in {"completed", "cancelled"}:
                            break
                        time.sleep(0.5)

    with sqlite3.connect(db_path) as conn:
        face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        cluster_count = conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]
        video_paths = [row[0] for row in conn.execute("SELECT last_known_filepath FROM scanned_files").fetchall()]

    print(f"Faces: {face_count}")
    print(f"Clusters: {cluster_count}")
    for path in video_paths:
        try:
            comment = ffmpeg.probe(path).get("format", {}).get("tags", {}).get("comment", "")
        except Exception:
            comment = ""
        print(f"{os.path.basename(path)} comment: {comment}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline test")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="test_vids",
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Temporary working directory to store database and thumbnails",
    )
    args = parser.parse_args()

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="indexium_e2e_")
    os.makedirs(work_dir, exist_ok=True)
    video_dir = os.path.join(work_dir, "videos")
    shutil.copytree(args.input_dir, video_dir, dirs_exist_ok=True)

    run_pipeline(os.path.abspath(video_dir), os.path.abspath(work_dir))


if __name__ == "__main__":
    main()
