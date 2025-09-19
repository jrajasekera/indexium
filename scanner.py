import json
import os
import pickle
import signal
import sqlite3
import logging
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Optional

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import ffmpeg

from config import Config
from signal_handler import SignalHandler
from util import get_file_hash

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = Config()

# --- CONFIGURATION ---
VIDEO_DIRECTORY = config.VIDEO_DIR
DATABASE_FILE = config.DATABASE_FILE
FRAME_SKIP = config.FRAME_SKIP
CPU_CORES_TO_USE = config.CPU_CORES
SAVE_CHUNK_SIZE = config.SAVE_CHUNK_SIZE

def get_file_hash_with_path(filepath):
    """Return the given filepath alongside its computed hash."""
    return filepath, get_file_hash(filepath)

def validate_video_file(video_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validates a video file using ffprobe to check for corruption and basic properties.
    Returns (is_valid, error_message).
    """
    try:
        probe = ffmpeg.probe(video_path, loglevel="error")
        format_info = probe.get('format', {})
        duration = float(format_info.get('duration', 0))

        # Check for basic validity
        if duration <= 0:
            return False, "Video has zero or negative duration"

        # Check for video streams
        video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
        if not video_streams:
            return False, "No video streams found"

        # Check for corrupted streams (basic check)
        for stream in video_streams:
            if stream.get('codec_name') is None:
                return False, "Corrupted video stream detected"

        return True, None

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        return False, f"FFprobe error: {error_msg}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def save_thumbnail(face_id, video_path, frame_number, location_str):
    """Extracts and saves a thumbnail for a face."""
    os.makedirs(config.THUMBNAIL_DIR, exist_ok=True)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path} for thumbnail")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.error(f"Could not read frame {frame_number} from {video_path} for thumbnail")
            return

        top, right, bottom, left = map(int, location_str.split(','))
        face_img = frame[top:bottom, left:right]
        thumb_path = os.path.join(config.THUMBNAIL_DIR, f"{face_id}.jpg")
        cv2.imwrite(thumb_path, face_img)
        logger.debug(f"Thumbnail saved: {thumb_path}")
    except Exception as e:
        logger.error(f"Failed to create thumbnail for {video_path}: {e}")

def cleanup_failed_thumbnails():
    """Removes thumbnail files that don't have corresponding face records."""
    logger.info("Starting cleanup of orphaned thumbnails...")
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            # Get all face IDs that exist in the database
            existing_face_ids = {row[0] for row in cursor.execute("SELECT id FROM faces")}

        thumbnail_dir = config.THUMBNAIL_DIR
        if not os.path.exists(thumbnail_dir):
            return

        cleaned_count = 0
        for filename in os.listdir(thumbnail_dir):
            if filename.endswith('.jpg'):
                try:
                    face_id = int(filename[:-4])  # Remove .jpg extension
                    if face_id not in existing_face_ids:
                        thumb_path = os.path.join(thumbnail_dir, filename)
                        os.remove(thumb_path)
                        cleaned_count += 1
                except (ValueError, OSError) as e:
                    logger.warning(f"Error processing thumbnail {filename}: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} orphaned thumbnail files")
    except Exception as e:
        logger.error(f"Error during thumbnail cleanup: {e}")

def retry_failed_videos():
    """Attempts to reprocess videos that previously failed."""
    logger.info("Starting retry of failed videos...")
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            # Get failed videos
            failed_videos = cursor.execute(
                "SELECT file_hash, last_known_filepath FROM scanned_files WHERE processing_status = 'failed'"
            ).fetchall()

        if not failed_videos:
            logger.info("No failed videos to retry")
            return

        logger.info(f"Found {len(failed_videos)} failed videos to retry")

        successful_retries = 0
        for file_hash, video_path in failed_videos:
            if not os.path.exists(video_path):
                logger.warning(f"Video no longer exists: {video_path}")
                continue

            # Process the video
            result = process_video_job((video_path, file_hash))
            _, _, success, faces_list, error_message = result

            if success:
                # Update database with successful retry
                with sqlite3.connect(DATABASE_FILE) as conn:
                    cursor = conn.cursor()
                    face_count = len(faces_list)
                    manual_status = 'pending' if face_count == 0 else 'not_required'
                    cursor.execute(
                        '''
                        UPDATE scanned_files
                        SET processing_status = ?,
                            error_message = NULL,
                            last_attempt = CURRENT_TIMESTAMP,
                            face_count = ?,
                            manual_review_status = CASE
                                WHEN ? = 'pending' THEN
                                    CASE
                                        WHEN manual_review_status IN ('done', 'no_people') THEN manual_review_status
                                        ELSE 'pending'
                                    END
                                ELSE ?
                            END
                        WHERE file_hash = ?
                        ''',
                        ('completed', face_count, manual_status, manual_status, file_hash)
                    )
                    # Insert faces
                    for face_data in faces_list:
                        cursor.execute(
                            'INSERT INTO faces (file_hash, frame_number, face_location, face_encoding) VALUES (?, ?, ?, ?)',
                            face_data
                        )
                    conn.commit()
                successful_retries += 1
                logger.info(f"Successfully retried: {video_path}")
            else:
                # Update with new error message
                with sqlite3.connect(DATABASE_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'UPDATE scanned_files SET error_message = ?, last_attempt = CURRENT_TIMESTAMP WHERE file_hash = ?',
                        (error_message, file_hash)
                    )
                    conn.commit()
                logger.warning(f"Retry failed for {video_path}: {error_message}")

        logger.info(f"Retry complete. Successfully retried {successful_retries}/{len(failed_videos)} videos")

    except Exception as e:
        logger.error(f"Error during retry process: {e}")


def setup_database():
    """Initializes the SQLite database and creates the necessary tables with schema migration."""
    logger.info("Setting up database...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scanned_files (
                file_hash TEXT PRIMARY KEY,
                last_known_filepath TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                face_location TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                cluster_id INTEGER DEFAULT NULL,
                person_name TEXT DEFAULT NULL,
                suggested_person_name TEXT DEFAULT NULL,
                suggested_confidence REAL DEFAULT NULL,
                suggestion_status TEXT DEFAULT NULL,
                suggested_candidates TEXT DEFAULT NULL,
                FOREIGN KEY (file_hash) REFERENCES scanned_files (file_hash)
            )
        ''')

        cursor.execute("PRAGMA table_info(faces)")
        face_columns = {row[1] for row in cursor.fetchall()}
        if 'suggested_person_name' not in face_columns:
            logger.info("Adding suggested_person_name column to faces table...")
            cursor.execute("ALTER TABLE faces ADD COLUMN suggested_person_name TEXT DEFAULT NULL")
        if 'suggested_confidence' not in face_columns:
            logger.info("Adding suggested_confidence column to faces table...")
            cursor.execute("ALTER TABLE faces ADD COLUMN suggested_confidence REAL DEFAULT NULL")
        if 'suggestion_status' not in face_columns:
            logger.info("Adding suggestion_status column to faces table...")
            cursor.execute("ALTER TABLE faces ADD COLUMN suggestion_status TEXT DEFAULT NULL")
        if 'suggested_candidates' not in face_columns:
            logger.info("Adding suggested_candidates column to faces table...")
            cursor.execute("ALTER TABLE faces ADD COLUMN suggested_candidates TEXT DEFAULT NULL")

        # Check and add new columns to scanned_files table
        cursor.execute("PRAGMA table_info(scanned_files)")
        columns = {row[1] for row in cursor.fetchall()}

        if 'processing_status' not in columns:
            logger.info("Adding processing_status column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN processing_status TEXT DEFAULT 'pending'")

        if 'error_message' not in columns:
            logger.info("Adding error_message column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN error_message TEXT DEFAULT NULL")

        if 'last_attempt' not in columns:
            logger.info("Adding last_attempt column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN last_attempt TIMESTAMP")
            # Update existing rows with current timestamp
            cursor.execute("UPDATE scanned_files SET last_attempt = CURRENT_TIMESTAMP WHERE last_attempt IS NULL")

        if 'face_count' not in columns:
            logger.info("Adding face_count column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN face_count INTEGER DEFAULT NULL")

        if 'manual_review_status' not in columns:
            logger.info("Adding manual_review_status column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN manual_review_status TEXT DEFAULT 'not_required'")

        if 'sample_seed' not in columns:
            logger.info("Adding sample_seed column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN sample_seed INTEGER DEFAULT NULL")

        # Add indexes for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_file_hash ON faces (file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces (cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_status ON scanned_files (processing_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_manual_status ON scanned_files (manual_review_status)')

        # Add composite indexes for frequently queried combinations
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_person_cluster ON faces (person_name, cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_cluster_person ON faces (cluster_id, person_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_file_person ON faces (file_hash, person_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces (person_name, id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_suggestion ON faces (cluster_id, suggested_person_name)')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                person_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_hash, person_name),
                FOREIGN KEY (file_hash) REFERENCES scanned_files (file_hash)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_people_file_hash ON video_people (file_hash)')

        # Backfill counts and manual review status for legacy records
        cursor.execute(
            '''
            UPDATE scanned_files
            SET face_count = (
                SELECT COUNT(*) FROM faces WHERE faces.file_hash = scanned_files.file_hash
            )
            WHERE face_count IS NULL
            '''
        )
        cursor.execute(
            '''
            UPDATE scanned_files
            SET manual_review_status = 'pending'
            WHERE manual_review_status = 'not_required'
              AND (
                  SELECT COUNT(*) FROM faces WHERE faces.file_hash = scanned_files.file_hash
              ) = 0
            '''
        )

        conn.commit()
    logger.info("Database setup complete.")


def process_video_job(job_data):
    """
    Worker function to process a single video file with enhanced error handling.
    Accepts a tuple (video_path, file_hash).
    Returns (file_hash, video_path, success, faces_list, error_message).
    """
    video_path, file_hash = job_data
    logger.info(f"Processing video: {video_path}")

    # Validate video file first
    is_valid, validation_error = validate_video_file(video_path)
    if not is_valid:
        error_msg = f"Video validation failed: {validation_error}"
        logger.error(f"Validation failed for {video_path}: {error_msg}")
        return (file_hash, video_path, False, [], error_msg)

    faces_found_in_video = []
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                error_msg = "Could not open video file with OpenCV"
                logger.error(f"OpenCV error for {video_path}: {error_msg}")
                return (file_hash, video_path, False, [], error_msg)

            frame_count = 0
            corrupted_frames = 0
            max_corrupted_frames = 10  # Allow some corrupted frames

            while True:
                try:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    if frame_count % FRAME_SKIP == 0:
                        # Check if frame is valid
                        if frame is None or frame.size == 0:
                            corrupted_frames += 1
                            if corrupted_frames > max_corrupted_frames:
                                error_msg = f"Too many corrupted frames ({corrupted_frames})"
                                logger.error(f"Frame corruption in {video_path}: {error_msg}")
                                video_capture.release()
                                return (file_hash, video_path, False, [], error_msg)
                            continue

                        # Resize frame for faster processing
                        try:
                            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            # Convert from BGR (OpenCV) to RGB (face_recognition)
                            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        except cv2.error as e:
                            corrupted_frames += 1
                            logger.warning(f"Frame processing error at frame {frame_count}: {e}")
                            continue

                        try:
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
                        except Exception as e:
                            logger.warning(f"Face detection error at frame {frame_count}: {e}")
                            # Continue processing other frames

                    frame_count += 1

                except Exception as e:
                    corrupted_frames += 1
                    logger.warning(f"Frame read error at frame {frame_count}: {e}")
                    if corrupted_frames > max_corrupted_frames:
                        break

            video_capture.release()
            logger.info(f"Successfully processed {video_path}. Found {len(faces_found_in_video)} faces.")
            return (file_hash, video_path, True, faces_found_in_video, None)

        except cv2.error as e:
            error_msg = f"OpenCV error: {str(e)}"
            logger.error(f"OpenCV error processing {video_path}: {error_msg}")
        except MemoryError as e:
            error_msg = f"Memory error: {str(e)}"
            logger.error(f"Memory error processing {video_path}: {error_msg}")
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error processing {video_path}: {error_msg}")

        retry_count += 1
        if retry_count < max_retries:
            logger.info(f"Retrying {video_path} (attempt {retry_count + 1}/{max_retries})")
            # Clean up any resources
            if 'video_capture' in locals():
                video_capture.release()

    error_msg = f"Failed after {max_retries} attempts"
    logger.error(f"Final failure for {video_path}: {error_msg}")
    return (file_hash, video_path, False, [], error_msg)


def write_data_to_db(face_data, scanned_files_info, failed_files_info=None):
    """Writes a chunk of collected data to the SQLite database and saves thumbnails."""
    if not face_data and not scanned_files_info and not failed_files_info:
        return

    success_count = len(scanned_files_info) if scanned_files_info else 0
    failed_count = len(failed_files_info) if failed_files_info else 0
    faces_count = len(face_data) if face_data else 0

    logger.info(f"Saving progress: {success_count} successful videos, {failed_count} failed videos, {faces_count} faces")

    try:
        with sqlite3.connect(DATABASE_FILE, timeout=30) as conn:
            cursor = conn.cursor()

            # Map hashes to paths for thumbnail generation
            file_map = {h: p for h, p, _ in scanned_files_info} if scanned_files_info else {}

            # Update successful files
            if scanned_files_info:
                upsert_rows = []
                for file_hash, path, face_count in scanned_files_info:
                    manual_status = 'pending' if face_count == 0 else 'not_required'
                    upsert_rows.append((file_hash, path, 'completed', face_count, manual_status))

                cursor.executemany(
                    '''
                    INSERT INTO scanned_files (
                        file_hash,
                        last_known_filepath,
                        processing_status,
                        error_message,
                        last_attempt,
                        face_count,
                        manual_review_status
                    ) VALUES (?, ?, ?, NULL, CURRENT_TIMESTAMP, ?, ?)
                    ON CONFLICT(file_hash) DO UPDATE SET
                        last_known_filepath = excluded.last_known_filepath,
                        processing_status = excluded.processing_status,
                        error_message = NULL,
                        last_attempt = CURRENT_TIMESTAMP,
                        face_count = excluded.face_count,
                        manual_review_status = CASE
                            WHEN excluded.manual_review_status = 'pending' THEN
                                CASE
                                    WHEN scanned_files.manual_review_status IN ('done', 'no_people') THEN scanned_files.manual_review_status
                                    ELSE 'pending'
                                END
                            WHEN excluded.manual_review_status IS NULL THEN scanned_files.manual_review_status
                            ELSE excluded.manual_review_status
                        END
                    ''',
                    upsert_rows,
                )

            # Update failed files
            if failed_files_info:
                cursor.executemany(
                    '''
                    INSERT INTO scanned_files (
                        file_hash,
                        last_known_filepath,
                        processing_status,
                        error_message,
                        last_attempt,
                        face_count,
                        manual_review_status
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, NULL, 'not_required')
                    ON CONFLICT(file_hash) DO UPDATE SET
                        last_known_filepath = excluded.last_known_filepath,
                        processing_status = excluded.processing_status,
                        error_message = excluded.error_message,
                        last_attempt = CURRENT_TIMESTAMP
                    ''',
                    [(h, p, 'failed', err) for h, p, err in failed_files_info],
                )

            # Insert faces
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
        logger.info("Save complete.")
    except Exception as e:
        logger.error(f"DATABASE ERROR during save: {e}")


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

    # Determine the number of processes for hashing
    num_hashing_processes = CPU_CORES_TO_USE if CPU_CORES_TO_USE is not None else cpu_count()
    print(f"Hashing files using {num_hashing_processes} processes...")

    # Hash files in parallel
    with Pool(processes=num_hashing_processes) as pool:
        # Create a list of filepaths that need hashing
        filepaths_to_hash = all_video_files
        # Use imap_unordered for progress reporting, though map would also work
        hashed_files = {}
        total_files = len(filepaths_to_hash)
        processed_count = 0

        results_iterator = pool.imap_unordered(get_file_hash_with_path, filepaths_to_hash)

        for filepath, file_hash in results_iterator:
            if handler.shutdown_requested:
                print("[Main] Shutdown detected during file hashing. Stopping.")
                break

            processed_count += 1
            print(f"[{processed_count}/{total_files}] Hashed: {filepath} | Hash: {file_hash if file_hash else 'FAILED'}")

            if file_hash:
                hashed_files[filepath] = file_hash

    if handler.shutdown_requested:
        print("[Main] Hashing process stopped.")
        return

    jobs_to_process = []
    for filepath, file_hash in hashed_files.items():
        if file_hash not in scanned_hashes:
            try:
                file_size = os.path.getsize(filepath)
                jobs_to_process.append((filepath, file_hash, file_size))
            except OSError as e:
                print(f"Warning: Could not get size for {filepath}: {e}")
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
    pending_failed_info = []
    total_processed = 0
    total_successful = 0
    total_failed = 0

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered to get results as they are completed
        results_iterator = pool.imap_unordered(process_video_job, jobs_to_process)

        for result in results_iterator:
            if handler.shutdown_requested:
                break  # Exit the loop if shutdown is requested

            file_hash, video_path, success, faces_list, error_message = result
            total_processed += 1

            if success:
                total_successful += 1
                face_count = len(faces_list)
                pending_files_info.append((file_hash, video_path, face_count))
                pending_faces.extend(faces_list)
            else:
                total_failed += 1
                logger.warning(f"Failed to process {video_path}: {error_message}")
                pending_failed_info.append((file_hash, video_path, error_message))

            # Save to DB when chunk size is reached
            if len(pending_files_info) + len(pending_failed_info) >= SAVE_CHUNK_SIZE:
                write_data_to_db(pending_faces, pending_files_info, pending_failed_info)
                pending_faces, pending_files_info, pending_failed_info = [], [], []  # Reset chunks

    # After the loop (or on shutdown), save any remaining data
    if pending_faces or pending_files_info or pending_failed_info:
        logger.info("Performing final save...")
        write_data_to_db(pending_faces, pending_files_info, pending_failed_info)

    # Print summary
    logger.info(f"Video scanning complete. Processed: {total_processed}, Successful: {total_successful}, Failed: {total_failed}")
    if total_failed > 0:
        logger.warning(f"{total_failed} videos failed processing. Check the database for error details.")


def classify_new_faces():
    """Assigns person names to untagged faces based on known people."""
    print("Starting automatic face classification...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()

        # Fetch encodings of already named people
        rows = cursor.execute(
            "SELECT person_name, face_encoding FROM faces WHERE person_name IS NOT NULL AND person_name != ?",
            ("Unknown",)
        ).fetchall()

        if not rows:
            print("No named faces available for classification.")
            return

        name_to_encs = {}
        for name, enc_blob in rows:
            enc = pickle.loads(enc_blob)
            name_to_encs.setdefault(name, []).append(enc)

        base_threshold = config.AUTO_CLASSIFY_THRESHOLD
        person_models = {}
        for name, encs in name_to_encs.items():
            enc_array = np.stack(encs)
            centroid = np.mean(enc_array, axis=0)
            if len(encs) > 1:
                centroid_dists = np.linalg.norm(enc_array - centroid, axis=1)
                mean_dist = float(np.mean(centroid_dists))
                std_dist = float(np.std(centroid_dists))
                adaptive = max(mean_dist + std_dist * 0.75, base_threshold * 0.5)
            else:
                adaptive = base_threshold * 0.6
            adaptive = min(adaptive, base_threshold)
            adaptive = max(adaptive, base_threshold * 0.4)
            person_models[name] = {
                "encodings": enc_array,
                "centroid": centroid,
                "threshold": adaptive,
            }

        # Fetch faces without a name
        rows = cursor.execute(
            "SELECT id, face_encoding, suggested_person_name, suggestion_status, suggested_confidence, cluster_id "
            "FROM faces WHERE person_name IS NULL"
        ).fetchall()

        if not rows:
            print("No unnamed faces to classify.")
            return

        updates = []
        clears = []
        candidate_updates = []
        for face_id, enc_blob, current_suggestion, current_status, current_confidence, cluster_id in rows:
            enc = pickle.loads(enc_blob)
            if cluster_id == -1:
                candidate_updates.append((None, face_id))
                continue
            if not person_models:
                candidate_updates.append((None, face_id))
                continue

            candidates = []
            for name, model in person_models.items():
                distances = np.linalg.norm(model["encodings"] - enc, axis=1)
                nearest = float(np.min(distances))
                centroid_dist = float(np.linalg.norm(enc - model["centroid"]))
                blended_score = (0.65 * nearest) + (0.35 * centroid_dist)
                candidates.append((name, nearest, centroid_dist, blended_score, model["threshold"]))

            if not candidates:
                candidate_updates.append((None, face_id))
                continue

            candidates.sort(key=lambda item: item[3])
            best_name, best_nearest, _, best_score, best_threshold = candidates[0]
            second_candidate = candidates[1] if len(candidates) > 1 else None

            top_candidates_payload = []
            for cand_name, cand_nearest, _, cand_score, cand_threshold in candidates[:5]:
                safe_threshold = cand_threshold if cand_threshold and cand_threshold > 0 else base_threshold
                cand_confidence = max(0.0, min(1.0, 1.0 - (cand_nearest / max(safe_threshold, 1e-6))))
                if cand_name == best_name and second_candidate:
                    separation = float(second_candidate[3] - best_score)
                    separation_factor = max(0.0, min(1.0, separation / 0.2))
                    cand_confidence *= max(separation_factor, 0.15)
                top_candidates_payload.append({
                    "name": cand_name,
                    "confidence": round(cand_confidence, 4)
                })

            candidate_json = json.dumps(top_candidates_payload) if top_candidates_payload else None
            candidate_updates.append((candidate_json, face_id))

            if not top_candidates_payload:
                continue

            confidence = top_candidates_payload[0]["confidence"]

            # Skip updating if the same suggestion was explicitly rejected
            if current_status == 'rejected' and current_suggestion == best_name:
                continue

            accept_suggestion = (
                best_nearest <= base_threshold
                and best_nearest <= best_threshold
                and confidence >= 0.6
            )

            if accept_suggestion:
                should_update = (
                    current_suggestion != best_name
                    or current_confidence is None
                    or confidence > (current_confidence or 0)
                    or current_status not in ('pending', None)
                )
                if should_update:
                    updates.append((best_name, confidence, 'pending', candidate_json, face_id))
            else:
                if current_status != 'rejected' and (
                    current_suggestion is not None
                    or current_confidence is not None
                    or current_status is not None
                ):
                    clears.append((face_id,))

        commit_needed = False

        if candidate_updates:
            cursor.executemany(
                "UPDATE faces SET suggested_candidates = ? WHERE id = ?",
                candidate_updates
            )
            commit_needed = True

        if updates:
            cursor.executemany(
                "UPDATE faces SET suggested_person_name = ?, suggested_confidence = ?, suggestion_status = ?, suggested_candidates = COALESCE(?, suggested_candidates) WHERE id = ?",
                updates
            )
            commit_needed = True
            print(f"Added high-confidence suggestions for {len(updates)} faces.")
        if clears:
            cursor.executemany(
                "UPDATE faces SET suggested_person_name = NULL, suggested_confidence = NULL, suggestion_status = NULL WHERE id = ?",
                clears
            )
            commit_needed = True
            print(f"Cleared low-confidence suggestions for {len(clears)} faces.")

        if commit_needed:
            conn.commit()

        if not updates and not clears:
            print("No faces matched existing people within threshold.")


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
    import sys

    # Set up the signal handler for graceful shutdown
    signalHandler = SignalHandler()
    signal.signal(signal.SIGINT, signalHandler)  # Catches Ctrl+C
    signal.signal(signal.SIGTERM, signalHandler)  # Catches standard termination signal

    # Check command line arguments for special operations
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "retry":
            setup_database()
            retry_failed_videos()
            sys.exit(0)
        elif command == "cleanup":
            setup_database()
            cleanup_failed_thumbnails()
            sys.exit(0)
        else:
            print("Usage: python scanner.py [retry|cleanup]")
            sys.exit(1)

    setup_database()
    scan_videos_parallel(signalHandler)

    # Only run clustering if the process wasn't interrupted
    if not signalHandler.shutdown_requested:
        classify_new_faces()
        cluster_faces()
        # Clean up orphaned thumbnails after processing
        cleanup_failed_thumbnails()
    else:
        print("[Main] Clustering skipped due to script interruption.")

    print("[Main] Program finished.")
