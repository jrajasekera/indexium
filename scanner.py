import os
import pickle
import signal
import sqlite3
from multiprocessing import Pool, cpu_count
import logging

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN

from config import Config
from signal_handler import SignalHandler
from util import get_file_hash

logger = logging.getLogger(__name__)
config = Config()

# --- CONFIGURATION ---
VIDEO_DIRECTORY = config.VIDEO_DIR
DATABASE_FILE = config.DATABASE_FILE
FRAME_SKIP = config.FRAME_SKIP
CPU_CORES_TO_USE = config.CPU_CORES
SAVE_CHUNK_SIZE = config.SAVE_CHUNK_SIZE

def save_thumbnail(face_id, video_path, frame_number, location_str):
    """Extracts and saves a thumbnail for a face."""
    os.makedirs(config.THUMBNAIL_DIR, exist_ok=True)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("  - [Thumb Error] Could not open video %s", video_path)
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.warning(
                "  - [Thumb Error] Could not read frame %s from %s",
                frame_number,
                video_path,
            )
            return

        top, right, bottom, left = map(int, location_str.split(','))
        face_img = frame[top:bottom, left:right]
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{face_id}.jpg")
        cv2.imwrite(thumb_path, face_img)
    except Exception as e:
        logger.warning(
            "  - [Thumb Error] Failed to create thumbnail for %s: %s",
            video_path,
            e,
        )


def setup_database():
    """Initializes the SQLite database and creates the necessary tables."""
    logger.info("Setting up database...")
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
    logger.info("Database setup complete.")


def process_video_job(job_data):
    """
    Worker function to process a single video file.
    Accepts a tuple (video_path, file_hash).
    Returns its findings to the main process.
    """
    video_path, file_hash = job_data
    logger.info("[Worker] Processing: %s", video_path)
    faces_found_in_video = []
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            logger.warning(
                "  - [Worker Error] Could not open video file: %s", video_path
            )
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
        logger.info(
            "[Worker] Finished %s. Found %s faces.",
            video_path,
            len(faces_found_in_video),
        )
        return (file_hash, video_path, True, faces_found_in_video)
    except Exception as e:
        logger.error(
            "  - [Worker Error] An error occurred while processing %s: %s",
            video_path,
            e,
        )
        return (file_hash, video_path, False, [])


def write_data_to_db(face_data, scanned_files_info):
    """Writes a chunk of collected data to the SQLite database and saves thumbnails."""
    if not face_data and not scanned_files_info:
        return
    logger.info(
        "[Main] Saving progress for %s videos and %s faces...",
        len(scanned_files_info),
        len(face_data),
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
        logger.info("[Main] Save complete.")
    except Exception as e:
        logger.error("[Main] DATABASE ERROR during save: %s", e)


def scan_videos_parallel(handler):
    """
    Scans videos in parallel, collecting results and saving them in chunks.
    Identifies videos by hash to avoid re-processing moved/renamed files.
    """
    logger.info("Starting parallel video scan...")
    all_video_files = [os.path.join(r, f) for r, _, fs in os.walk(VIDEO_DIRECTORY) for f in fs if
                       f.lower().endswith(('.mp4', '.mkv', '.mov', '.avi'))]

    with sqlite3.connect(DATABASE_FILE) as conn:
        scanned_hashes = {row[0] for row in conn.execute("SELECT file_hash FROM scanned_files")}

    logger.info(
        "Found %s video files. Identifying new or changed files by hashing. This may take a moment...",
        len(all_video_files),
    )

    # Determine the number of processes for hashing
    num_hashing_processes = CPU_CORES_TO_USE if CPU_CORES_TO_USE is not None else cpu_count()
    logger.info("Hashing files using %s processes...", num_hashing_processes)

    # Hash files in parallel
    with Pool(processes=num_hashing_processes) as pool:
        # Create a list of filepaths that need hashing
        filepaths_to_hash = all_video_files
        # Use imap_unordered for progress reporting, though map would also work
        hashed_files = {}
        total_files = len(filepaths_to_hash)
        processed_count = 0

        results_iterator = pool.imap_unordered(get_file_hash, filepaths_to_hash)

        for filepath, file_hash in zip(filepaths_to_hash, results_iterator):
            if handler.shutdown_requested:
                logger.warning("[Main] Shutdown detected during file hashing. Stopping.")
                break

            processed_count += 1
            logger.info(
                "[%s/%s] Hashed: %s | Hash: %s",
                processed_count,
                total_files,
                filepath,
                file_hash if file_hash else "FAILED",
            )

            if file_hash:
                hashed_files[filepath] = file_hash

    if handler.shutdown_requested:
        logger.info("[Main] Hashing process stopped.")
        return

    jobs_to_process = []
    for filepath, file_hash in hashed_files.items():
        if file_hash not in scanned_hashes:
            try:
                file_size = os.path.getsize(filepath)
                jobs_to_process.append((filepath, file_hash, file_size))
            except OSError as e:
                logger.warning("Warning: Could not get size for %s: %s", filepath, e)
                jobs_to_process.append((filepath, file_hash, 0))

    if not jobs_to_process:
        logger.info("No new videos to scan.")
        return

    # Sort jobs by file size (smallest first)
    jobs_to_process.sort(key=lambda x: x[2])  # Sort by the file size (third element)
    logger.info(
        "Found %s new videos to process, sorted by size (smallest first).",
        len(jobs_to_process),
    )

    # Convert back to (filepath, file_hash) tuples for the worker function
    jobs_to_process = [(filepath, file_hash) for filepath, file_hash, _ in jobs_to_process]

    num_processes = CPU_CORES_TO_USE if CPU_CORES_TO_USE is not None else cpu_count()
    logger.info(
        "Creating a pool of %s worker processes. Press Ctrl+C to stop gracefully.",
        num_processes,
    )

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
                logger.warning("[Main] Failed to process: %s", video_path)

            # Save to DB when chunk size is reached
            if len(pending_files_info) >= SAVE_CHUNK_SIZE:
                write_data_to_db(pending_faces, pending_files_info)
                pending_faces, pending_files_info = [], []  # Reset chunks

    # After the loop (or on shutdown), save any remaining data
    logger.info("[Main] All workers finished or shutdown initiated. Performing final save.")
    write_data_to_db(pending_faces, pending_files_info)
    logger.info("Parallel video scanning complete.")


def classify_new_faces():
    """Assigns person names to untagged faces based on known people."""
    logger.info("Starting automatic face classification...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()

        # Fetch encodings of already named people
        rows = cursor.execute(
            "SELECT person_name, face_encoding FROM faces WHERE person_name IS NOT NULL"
        ).fetchall()

        if not rows:
            logger.info("No named faces available for classification.")
            return

        name_to_encs = {}
        for name, enc_blob in rows:
            enc = pickle.loads(enc_blob)
            name_to_encs.setdefault(name, []).append(enc)

        centroids = {name: np.mean(encs, axis=0) for name, encs in name_to_encs.items()}

        # Fetch faces without a name
        rows = cursor.execute(
            "SELECT id, face_encoding FROM faces WHERE person_name IS NULL"
        ).fetchall()

        if not rows:
            logger.info("No unnamed faces to classify.")
            return

        threshold = config.AUTO_CLASSIFY_THRESHOLD
        updates = []
        for face_id, enc_blob in rows:
            enc = pickle.loads(enc_blob)
            best_name = None
            best_dist = None
            for name, centroid in centroids.items():
                dist = np.linalg.norm(enc - centroid)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_name = name
            if best_dist is not None and best_dist <= threshold:
                updates.append((best_name, face_id))

        if updates:
            cursor.executemany(
                "UPDATE faces SET person_name = ? WHERE id = ?", updates
            )
            conn.commit()
            logger.info(
                "Automatically assigned %s faces to existing people.",
                len(updates),
            )
        else:
            logger.info("No faces matched existing people within threshold.")


def cluster_faces():
    """
    Fetches all face encodings from the database and uses DBSCAN to group them.
    """
    logger.info("Starting face clustering...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        # Fetch all faces that haven't been assigned a person name yet
        rows = cursor.execute(
            "SELECT id, face_encoding, cluster_id FROM faces WHERE person_name IS NULL"
        ).fetchall()
        if not rows:
            logger.info("No unnamed faces to cluster.")
            return

        logger.info("Found %s unnamed faces to cluster.", len(rows))
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

        logger.info(
            "Clustering complete. Found %s unique groups (including noise).",
            len(np.unique(clt.labels_)),
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
            logger.info("Updated %s faces with new cluster IDs.", len(updates))
        else:
            logger.info("No new clusters were formed.")

    logger.info("Face clustering complete.")


if __name__ == "__main__":
    # Set up the signal handler for graceful shutdown
    signalHandler = SignalHandler()
    signal.signal(signal.SIGINT, signalHandler)  # Catches Ctrl+C
    signal.signal(signal.SIGTERM, signalHandler)  # Catches standard termination signal

    setup_database()
    scan_videos_parallel(signalHandler)

    # Only run clustering if the process wasn't interrupted
    if not signalHandler.shutdown_requested:
        classify_new_faces()
        cluster_faces()
    else:
        logger.info("[Main] Clustering skipped due to script interruption.")

    logger.info("[Main] Program finished.")
