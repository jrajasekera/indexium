import os
import pickle
import signal
import sqlite3
import sys
from multiprocessing import Pool, cpu_count

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN

# --- CONFIGURATION ---
# Get video directory from environment variable
VIDEO_DIRECTORY = os.environ.get("INDEXIUM_VIDEO_DIR")
if VIDEO_DIRECTORY is None:
    print("Error: INDEXIUM_VIDEO_DIR environment variable not set")
    print("Please set this variable to the directory containing your videos")
    sys.exit(1)

DATABASE_FILE = "video_faces.db"
# How many frames to skip between face scans. Higher is faster but less thorough.
FRAME_SKIP = 25
# Number of CPU cores to use for parallel processing.
# Set to None to use all available cores.
CPU_CORES_TO_USE = 8
# How many videos to process before saving results to the database.
# A smaller number means more frequent saves but slightly more overhead.
SAVE_CHUNK_SIZE = 10


# --- Graceful Shutdown Handler ---
class SignalHandler:
    """A class to handle shutdown signals gracefully."""

    def __init__(self):
        self.shutdown_requested = False

    def __call__(self, signum, frame):
        print(f"\n[Main] Shutdown signal {signum} received. Finishing current tasks and saving...")
        self.shutdown_requested = True


def setup_database():
    """Initializes the SQLite database and creates the necessary tables."""
    print("Setting up database...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scanned_files (
                filepath TEXT PRIMARY KEY
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_filepath TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                face_location TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                cluster_id INTEGER DEFAULT NULL,
                person_name TEXT DEFAULT NULL
            )
        ''')
        conn.commit()
    print("Database setup complete.")


def process_video(video_path):
    """
    Worker function to process a single video file.
    Returns its findings to the main process.
    """
    print(f"[Worker] Processing: {video_path}")
    faces_found_in_video = []
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"  - [Worker Error] Could not open video file: {video_path}")
            return (video_path, False, [])

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret: break
            if frame_count % FRAME_SKIP == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for location, encoding in zip(face_locations, face_encodings):
                    top, right, bottom, left = location
                    location_original = (top * 2, right * 2, bottom * 2, left * 2)
                    location_str = ",".join(map(str, location_original))
                    encoding_blob = pickle.dumps(encoding)
                    faces_found_in_video.append((video_path, frame_count, location_str, encoding_blob))
            frame_count += 1
        video_capture.release()
        print(f"[Worker] Finished {video_path}. Found {len(faces_found_in_video)} faces.")
        return (video_path, True, faces_found_in_video)
    except Exception as e:
        print(f"  - [Worker Error] An error occurred while processing {video_path}: {e}")
        return (video_path, False, [])


def write_data_to_db(face_data, scanned_files):
    """Writes a chunk of collected data to the SQLite database."""
    if not face_data and not scanned_files:
        return
    print(f"[Main] Saving progress for {len(scanned_files)} videos and {len(face_data)} faces...")
    try:
        with sqlite3.connect(DATABASE_FILE, timeout=30) as conn:
            cursor = conn.cursor()
            if face_data:
                cursor.executemany(
                    'INSERT INTO faces (video_filepath, frame_number, face_location, face_encoding) VALUES (?, ?, ?, ?)',
                    face_data
                )
            if scanned_files:
                cursor.executemany(
                    'INSERT INTO scanned_files (filepath) VALUES (?)',
                    scanned_files
                )
            conn.commit()
        print("[Main] Save complete.")
    except Exception as e:
        print(f"[Main] DATABASE ERROR during save: {e}")


def scan_videos_parallel(handler):
    """
    Scans videos in parallel, collecting results and saving them in chunks.
    """
    print("Starting parallel video scan...")
    all_video_files = [os.path.join(r, f) for r, _, fs in os.walk(VIDEO_DIRECTORY) for f in fs if
                       f.lower().endswith(('.mp4', '.mkv', '.mov', '.avi'))]

    with sqlite3.connect(DATABASE_FILE) as conn:
        scanned_files = {row[0] for row in conn.execute("SELECT filepath FROM scanned_files")}

    files_to_process = [f for f in all_video_files if f not in scanned_files]
    if not files_to_process:
        print("No new videos to scan.")
        return

    print(f"Found {len(files_to_process)} new videos to process out of {len(all_video_files)} total.")
    num_processes = CPU_CORES_TO_USE if CPU_CORES_TO_USE is not None else cpu_count()
    print(f"Creating a pool of {num_processes} worker processes. Press Ctrl+C to stop gracefully.")

    pending_faces = []
    pending_files = []

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered to get results as they are completed
        results_iterator = pool.imap_unordered(process_video, files_to_process)

        for video_path, success, faces_list in results_iterator:
            if handler.shutdown_requested:
                break  # Exit the loop if shutdown is requested

            if success:
                pending_files.append((video_path,))
                pending_faces.extend(faces_list)
            else:
                print(f"[Main] Failed to process: {video_path}")

            # Save to DB when chunk size is reached
            if len(pending_files) >= SAVE_CHUNK_SIZE:
                write_data_to_db(pending_faces, pending_files)
                pending_faces, pending_files = [], []  # Reset chunks

    # After the loop (or on shutdown), save any remaining data
    print("[Main] All workers finished or shutdown initiated. Performing final save.")
    write_data_to_db(pending_faces, pending_files)
    print("Parallel video scanning complete.")


def cluster_faces():
    """
    Fetches all face encodings from the database and uses DBSCAN to group them.
    """
    print("Starting face clustering...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        rows = cursor.execute("SELECT id, face_encoding FROM faces WHERE cluster_id IS NULL").fetchall()
        if not rows:
            print("No new faces to cluster.")
            return

        print(f"Found {len(rows)} new faces to cluster.")
        face_ids = [row[0] for row in rows]
        encodings = [pickle.loads(row[1]) for row in rows]

        clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=0.4, min_samples=5)
        clt.fit(encodings)

        max_cluster_id = cursor.execute("SELECT MAX(cluster_id) FROM faces").fetchone()[0] or 0
        label_offset = max_cluster_id + 1

        print(f"Clustering complete. Found {len(np.unique(clt.labels_))} unique groups (including noise).")
        updates = [(int(label) + label_offset if label != -1 else None, face_id) for face_id, label in
                   zip(face_ids, clt.labels_)]

        cursor.executemany("UPDATE faces SET cluster_id = ? WHERE id = ?", updates)
        conn.commit()
    print("Face clustering complete.")


if __name__ == "__main__":
    # Set up the signal handler
    handler = SignalHandler()
    signal.signal(signal.SIGINT, handler)  # Catches Ctrl+C
    signal.signal(signal.SIGTERM, handler)  # Catches standard termination signal

    setup_database()
    scan_videos_parallel(handler)

    # Only run clustering if the process wasn't interrupted
    if not handler.shutdown_requested:
        cluster_faces()
    else:
        print("[Main] Clustering skipped due to script interruption.")

    print("[Main] Program finished.")
