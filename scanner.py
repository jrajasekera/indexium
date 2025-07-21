import os
import pickle
import signal
import sqlite3
from multiprocessing import Pool, cpu_count

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN

from config import Config
from util import get_file_hash

config = Config()


def save_thumbnail(face_id, video_path, frame_number, location_str):
    """Extracts and saves a thumbnail for a face."""
    os.makedirs(config.THUMBNAIL_DIR, exist_ok=True)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  - [Thumb Error] Could not open video {video_path}")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  - [Thumb Error] Could not read frame {frame_number} from {video_path}")
            return

        top, right, bottom, left = map(int, location_str.split(','))
        face_img = frame[top:bottom, left:right]
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{face_id}.jpg")
        cv2.imwrite(thumb_path, face_img)
    except Exception as e:
        print(f"  - [Thumb Error] Failed to create thumbnail for {video_path}: {e}")

# --- CONFIGURATION ---
# Load configuration from environment variables
VIDEO_DIRECTORY = config.VIDEO_DIR
DATABASE_FILE = config.DATABASE_FILE
FRAME_SKIP = config.FRAME_SKIP
CPU_CORES_TO_USE = config.CPU_CORES
SAVE_CHUNK_SIZE = config.SAVE_CHUNK_SIZE


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
        # Use file hash as the primary key to uniquely identify files regardless of path
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scanned_files (
                file_hash TEXT PRIMARY KEY,
                last_known_filepath TEXT NOT NULL
            )
        ''')
        # Faces are linked to the file via its hash
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                face_location TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                cluster_id INTEGER DEFAULT NULL,
                person_name TEXT DEFAULT NULL,
                FOREIGN KEY (file_hash) REFERENCES scanned_files (file_hash)
            )
        ''')
        # Add indexes for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_file_hash ON faces (file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces (cluster_id)')
        conn.commit()
    print("Database setup complete.")


def process_video_job(job_data):
    """
    Worker function to process a single video file.
    Accepts a tuple (video_path, file_hash).
    Returns its findings to the main process.
    """
    video_path, file_hash = job_data
    print(f"[Worker] Processing: {video_path}")
    faces_found_in_video = []
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"  - [Worker Error] Could not open video file: {video_path}")
            return (file_hash, video_path, False, [])

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret: break
            if frame_count % FRAME_SKIP == 0:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                # Convert from BGR (OpenCV) to RGB (face_recognition)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(
                    rgb_frame, model=config.FACE_DETECTION_MODEL
                )
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for location, encoding in zip(face_locations, face_encodings):
                    # Scale coordinates back to original frame size
                    top, right, bottom, left = location
                    location_original = (top * 2, right * 2, bottom * 2, left * 2)
                    location_str = ",".join(map(str, location_original))
                    encoding_blob = pickle.dumps(encoding)
                    # Associate face with the file's hash, not its path
                    faces_found_in_video.append((file_hash, frame_count, location_str, encoding_blob))
            frame_count += 1
        video_capture.release()
        print(f"[Worker] Finished {video_path}. Found {len(faces_found_in_video)} faces.")
        return (file_hash, video_path, True, faces_found_in_video)
    except Exception as e:
        print(f"  - [Worker Error] An error occurred while processing {video_path}: {e}")
        return (file_hash, video_path, False, [])


def write_data_to_db(face_data, scanned_files_info):
    """Writes a chunk of collected data to the SQLite database and saves thumbnails."""
    if not face_data and not scanned_files_info:
        return
    print(
        f"[Main] Saving progress for {len(scanned_files_info)} videos and {len(face_data)} faces..."
    )
    try:
        with sqlite3.connect(DATABASE_FILE, timeout=30) as conn:
            cursor = conn.cursor()

            # Map hashes to paths for thumbnail generation
            file_map = {h: p for h, p in scanned_files_info}

            if scanned_files_info:
                cursor.executemany(
                    'REPLACE INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)',
                    scanned_files_info,
                )

            if face_data:
                for file_hash, frame_number, loc_str, enc_blob in face_data:
                    cursor.execute(
                        'INSERT INTO faces (file_hash, frame_number, face_location, face_encoding) VALUES (?, ?, ?, ?)',
                        (file_hash, frame_number, loc_str, enc_blob),
                    )
                    face_id = cursor.lastrowid
                    video_path = file_map.get(file_hash)
                    if video_path:
                        save_thumbnail(face_id, video_path, frame_number, loc_str)

            conn.commit()
        print("[Main] Save complete.")
    except Exception as e:
        print(f"[Main] DATABASE ERROR during save: {e}")


def scan_videos_parallel(handler):
    """
    Scans videos in parallel, collecting results and saving them in chunks.
    Identifies videos by hash to avoid re-processing moved/renamed files.
    """
    print("Starting parallel video scan...")
    all_video_files = [os.path.join(r, f) for r, _, fs in os.walk(VIDEO_DIRECTORY) for f in fs if
                       f.lower().endswith(('.mp4', '.mkv', '.mov', '.avi'))]

    with sqlite3.connect(DATABASE_FILE) as conn:
        scanned_hashes = {row[0] for row in conn.execute("SELECT file_hash FROM scanned_files")}

    print(
        f"Found {len(all_video_files)} video files. Identifying new or changed files by hashing. This may take a moment...")
    jobs_to_process = []

    # Counter for tracking progress
    processed_count = 0
    total_files = len(all_video_files)
    for filepath in all_video_files:
        if handler.shutdown_requested:
            print("[Main] Shutdown detected during file hashing. Stopping.")
            break

        processed_count += 1
        file_hash = get_file_hash(filepath)

        # Print progress information
        print(f"[{processed_count}/{total_files}] File: {filepath} | Hash: {file_hash if file_hash else 'FAILED'}")

        if file_hash and file_hash not in scanned_hashes:
            # Get file size for sorting
            try:
                file_size = os.path.getsize(filepath)
                jobs_to_process.append((filepath, file_hash, file_size))
            except OSError as e:
                print(f"Warning: Could not get size for {filepath}: {e}")
                # Add with size 0 if we can't get the actual size
                jobs_to_process.append((filepath, file_hash, 0))

    if not jobs_to_process:
        print("No new videos to scan.")
        return

    # Sort jobs by file size (smallest first)
    jobs_to_process.sort(key=lambda x: x[2])  # Sort by the file size (third element)
    print(f"Found {len(jobs_to_process)} new videos to process, sorted by size (smallest first).")

    # Convert back to (filepath, file_hash) tuples for the worker function
    jobs_to_process = [(filepath, file_hash) for filepath, file_hash, _ in jobs_to_process]

    num_processes = CPU_CORES_TO_USE if CPU_CORES_TO_USE is not None else cpu_count()
    print(f"Creating a pool of {num_processes} worker processes. Press Ctrl+C to stop gracefully.")

    pending_faces = []
    pending_files_info = []

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered to get results as they are completed
        results_iterator = pool.imap_unordered(process_video_job, jobs_to_process)

        for file_hash, video_path, success, faces_list in results_iterator:
            if handler.shutdown_requested:
                break  # Exit the loop if shutdown is requested

            if success:
                # We'll add the file to the DB with its hash and path
                pending_files_info.append((file_hash, video_path))
                pending_faces.extend(faces_list)
            else:
                print(f"[Main] Failed to process: {video_path}")

            # Save to DB when chunk size is reached
            if len(pending_files_info) >= SAVE_CHUNK_SIZE:
                write_data_to_db(pending_faces, pending_files_info)
                pending_faces, pending_files_info = [], []  # Reset chunks

    # After the loop (or on shutdown), save any remaining data
    print("[Main] All workers finished or shutdown initiated. Performing final save.")
    write_data_to_db(pending_faces, pending_files_info)
    print("Parallel video scanning complete.")


def cluster_faces():
    """
    Fetches all face encodings from the database and uses DBSCAN to group them.
    """
    print("Starting face clustering...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        # Fetch all faces that haven't been assigned a person name yet
        rows = cursor.execute(
            "SELECT id, face_encoding, cluster_id FROM faces WHERE person_name IS NULL"
        ).fetchall()
        if not rows:
            print("No unnamed faces to cluster.")
            return

        print(f"Found {len(rows)} unnamed faces to cluster.")
        face_ids = [row[0] for row in rows]
        encodings = [pickle.loads(row[1]) for row in rows]
        existing_cluster_ids = [row[2] for row in rows]

        # DBSCAN parameters:
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        clt = DBSCAN(
            metric="euclidean",
            n_jobs=-1,
            eps=config.DBSCAN_EPS,
            min_samples=config.DBSCAN_MIN_SAMPLES,
        )
        clt.fit(encodings)

        # Get the highest existing cluster_id to ensure new IDs are unique
        max_cluster_id_result = cursor.execute("SELECT MAX(cluster_id) FROM faces").fetchone()
        next_cluster_id = (
            max_cluster_id_result[0] + 1 if max_cluster_id_result and max_cluster_id_result[0] is not None else 1
        )

        print(
            f"Clustering complete. Found {len(np.unique(clt.labels_))} unique groups (including noise)."
        )

        # Map each cluster label to an existing or new cluster_id
        label_to_cluster = {}
        for label in set(clt.labels_):
            if label == -1:
                continue
            indices = [i for i, lbl in enumerate(clt.labels_) if lbl == label]
            existing_ids = [existing_cluster_ids[i] for i in indices if existing_cluster_ids[i] is not None]
            if existing_ids:
                # Reuse the most common existing cluster_id within this group
                cluster_id = max(set(existing_ids), key=existing_ids.count)
            else:
                cluster_id = next_cluster_id
                next_cluster_id += 1
            label_to_cluster[label] = cluster_id

        # Prepare updates for faces that belong to a cluster
        updates = []
        for idx, label in enumerate(clt.labels_):
            if label == -1:
                continue  # leave noise faces unclustered
            updates.append((label_to_cluster[label], face_ids[idx]))

        if updates:
            cursor.executemany("UPDATE faces SET cluster_id = ? WHERE id = ?", updates)
            conn.commit()
            print(f"Updated {len(updates)} faces with new cluster IDs.")
        else:
            print("No new clusters were formed.")

    print("Face clustering complete.")


if __name__ == "__main__":
    # Set up the signal handler for graceful shutdown
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
