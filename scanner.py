import json
import os
import pickle
import re
import signal
import sqlite3
import logging
import queue
import multiprocessing as mp
import shutil
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Optional, Dict, Any

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import ffmpeg

try:
    import easyocr
except ImportError:  # pragma: no cover - dependency optional at runtime
    easyocr = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional fallback dependency
    pytesseract = None

from config import Config
from signal_handler import SignalHandler
from util import get_file_hash
from text_utils import calculate_top_text_fragments

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = Config()

# --- PRE-FLIGHT CHECKS ---
def _require_ffprobe() -> None:
    """Ensure ffprobe is available before running video scans."""
    if shutil.which("ffprobe") is None:
        logger.error(
            "ffprobe not found in PATH. Install ffmpeg (includes ffprobe) and try again."
        )
        raise SystemExit(1)

# --- CONFIGURATION ---
VIDEO_DIRECTORY = config.VIDEO_DIR
DATABASE_FILE = config.DATABASE_FILE
FRAME_SKIP = config.FRAME_SKIP
CPU_CORES_TO_USE = config.CPU_CORES
SAVE_CHUNK_SIZE = config.SAVE_CHUNK_SIZE
OCR_ENABLED = config.OCR_ENABLED
OCR_ENGINE = (config.OCR_ENGINE or "auto").strip().lower()
if OCR_ENGINE not in {"easyocr", "tesseract", "auto"}:
    logger.warning("Unsupported OCR engine '%s'; defaulting to 'auto'.", OCR_ENGINE)
    OCR_ENGINE = "auto"
OCR_FRAME_INTERVAL = max(1, config.OCR_FRAME_INTERVAL)
OCR_MIN_CONFIDENCE = config.OCR_MIN_CONFIDENCE
OCR_MIN_TEXT_LENGTH = config.OCR_MIN_TEXT_LENGTH
OCR_MAX_TEXT_LENGTH = max(config.OCR_MIN_TEXT_LENGTH, config.OCR_MAX_TEXT_LENGTH)
OCR_MAX_RESULTS_PER_VIDEO = max(1, config.OCR_MAX_RESULTS_PER_VIDEO)
MIN_OCR_TEXT_LENGTH = max(4, OCR_MIN_TEXT_LENGTH)
TOP_FRAGMENT_COUNT = max(1, config.OCR_TOP_FRAGMENT_COUNT)

_ocr_reader = None
_easyocr_worker: Optional[tuple] = None  # (process, request_queue, response_queue)
ACTIVE_OCR_BACKEND: Optional[str] = None
_BACKEND_INITIALIZED = False

if not OCR_ENABLED:
    ACTIVE_OCR_BACKEND = 'disabled'


def _record_ocr_text(
    aggregator: Dict[str, Dict[str, Any]],
    cleaned_text: str,
    frame_index: int,
    timestamp_ms: Optional[int],
    confidence: Optional[float],
):
    """Merge a normalized OCR string into the aggregator respecting limits."""
    if not cleaned_text:
        return
    key = cleaned_text.lower()
    if key not in aggregator and len(aggregator) >= OCR_MAX_RESULTS_PER_VIDEO:
        return

    entry = aggregator.setdefault(
        key,
        {
            "raw_text": cleaned_text,
            "normalized_text": key,
            "confidence": confidence,
            "first_seen_frame": frame_index,
            "first_seen_timestamp_ms": timestamp_ms,
            "occurrence_count": 0,
        },
    )

    entry["occurrence_count"] += 1
    if confidence is not None:
        existing_conf = entry.get("confidence")
        if existing_conf is None or confidence > existing_conf:
            entry["confidence"] = confidence
            entry["raw_text"] = cleaned_text
    elif entry.get("confidence") is None:
        entry["raw_text"] = cleaned_text

    if frame_index < entry.get("first_seen_frame", frame_index):
        entry["first_seen_frame"] = frame_index
        entry["first_seen_timestamp_ms"] = timestamp_ms


def _easyocr_probe_worker(queue_obj, languages, use_gpu):  # pragma: no cover - subprocess helper
    try:
        easyocr.Reader(list(languages), gpu=use_gpu)
        queue_obj.put(("ok", None))
    except Exception as exc:  # noqa: BLE001
        queue_obj.put(("error", str(exc)))
    finally:
        queue_obj.close()


def _easyocr_worker_loop(request_q, response_q, languages, use_gpu):  # pragma: no cover - subprocess helper
    try:
        reader = easyocr.Reader(list(languages), gpu=use_gpu)
    except Exception as exc:  # noqa: BLE001
        response_q.put(("error", str(exc)))
        response_q.close()
        request_q.close()
        return

    response_q.put(("ready", None))

    while True:
        try:
            item = request_q.get()
        except (EOFError, OSError):
            break
        if item is None:
            break
        frame = item
        try:
            results = reader.readtext(frame, detail=1)
            response_q.put(("ok", results))
        except Exception as exc:  # noqa: BLE001
            response_q.put(("error", str(exc)))

    response_q.close()
    request_q.close()


def _refresh_worker_initializer(backend: str):  # pragma: no cover - runs in worker
    global ACTIVE_OCR_BACKEND, _BACKEND_INITIALIZED, OCR_ENABLED
    ACTIVE_OCR_BACKEND = backend
    OCR_ENABLED = backend not in (None, 'disabled')
    _BACKEND_INITIALIZED = True


def _probe_easyocr_support() -> bool:
    """Attempt to create an EasyOCR reader in an isolated process to avoid crashes."""
    if easyocr is None:
        return False

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_easyocr_probe_worker,
        args=(result_queue, config.OCR_LANGUAGES, config.OCR_USE_GPU),
    )
    process.start()
    process.join()

    status = "error"
    payload = None
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:  # Worker crashed before reporting (e.g., illegal instruction)
        status = "error"
    finally:
        result_queue.close()
        result_queue.join_thread()

    if process.exitcode != 0:
        logger.error(
            "EasyOCR probe process exited with code %s; disabling EasyOCR backend.",
            process.exitcode,
        )
        return False

    if status != "ok":
        message = payload or "unknown error"
        logger.error("EasyOCR probe failed: %s", message)
        return False

    return True


def _start_easyocr_worker() -> bool:
    """Spin up a dedicated EasyOCR worker process."""
    global _easyocr_worker
    if easyocr is None:
        return False

    ctx = mp.get_context("spawn")
    request_q = ctx.Queue()
    response_q = ctx.Queue()
    process = ctx.Process(
        target=_easyocr_worker_loop,
        args=(request_q, response_q, config.OCR_LANGUAGES, config.OCR_USE_GPU),
        daemon=True,
    )
    process.start()

    try:
        status, payload = response_q.get(timeout=60)
    except queue.Empty:
        status, payload = "error", "EasyOCR worker start timeout"

    if status != "ready":
        logger.error("EasyOCR worker failed to start: %s", payload)
        process.terminate()
        process.join(timeout=5)
        request_q.close()
        response_q.close()
        return False

    _easyocr_worker = (process, request_q, response_q)
    return True


def _stop_easyocr_worker():  # pragma: no cover - cleanup helper
    global _easyocr_worker
    if not _easyocr_worker:
        return
    process, request_q, response_q = _easyocr_worker
    try:
        request_q.put_nowait(None)
    except Exception:
        pass
    request_q.close()
    response_q.close()
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    _easyocr_worker = None


def _easyocr_request(frame: np.ndarray):
    """Send a frame to the EasyOCR worker and return raw results or None on failure."""
    global ACTIVE_OCR_BACKEND, _BACKEND_INITIALIZED
    if easyocr is None:
        return None

    worker = _easyocr_worker
    if worker is None or not worker[0].is_alive():
        if not _start_easyocr_worker():
            ACTIVE_OCR_BACKEND = 'tesseract'
            _BACKEND_INITIALIZED = True
            return None
        worker = _easyocr_worker

    process, request_q, response_q = worker

    try:
        request_q.put(frame, timeout=5)
        status, payload = response_q.get(timeout=120)
    except queue.Empty:
        logger.error("EasyOCR worker did not respond in time; falling back to Tesseract.")
        _stop_easyocr_worker()
        ACTIVE_OCR_BACKEND = 'tesseract'
        _BACKEND_INITIALIZED = True
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("EasyOCR worker communication error: %s", exc)
        _stop_easyocr_worker()
        ACTIVE_OCR_BACKEND = 'tesseract'
        return None

    if status == 'ok':
        return payload

    logger.warning("EasyOCR worker error: %s. Falling back to Tesseract.", payload)
    _stop_easyocr_worker()
    ACTIVE_OCR_BACKEND = 'tesseract'
    _BACKEND_INITIALIZED = True
    return None


def diagnose_ocr_environment():
    """Log diagnostics about the available OCR backends."""
    logger.info("--- OCR Environment Diagnostics ---")
    logger.info(
        "Configured engine=%s, languages=%s, use_gpu=%s, enabled=%s",
        OCR_ENGINE,
        ",".join(config.OCR_LANGUAGES),
        config.OCR_USE_GPU,
        OCR_ENABLED,
    )

    if easyocr is None:
        logger.warning("easyocr python package is not available in this environment.")
    else:
        logger.info("easyocr package found: %s", easyocr.__file__)
        logger.info("Probing EasyOCR reader startup...")
        if _probe_easyocr_support():
            logger.info("EasyOCR probe succeeded. The backend should be available.")
        else:
            logger.warning("EasyOCR probe failed. See logs above for details.")

    if pytesseract is None:
        logger.warning("pytesseract python package is not available. Tesseract backend disabled.")
    else:
        logger.info("pytesseract package found: %s", pytesseract.__file__)
        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract binary version detected: %s", version)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to execute tesseract binary via pytesseract: %s", exc)

    logger.info("--- End OCR Diagnostics ---")


def _store_fragments(cursor, file_hash: str, fragments: List[Dict[str, Any]]):
    cursor.execute('DELETE FROM video_text_fragments WHERE file_hash = ?', (file_hash,))
    if not fragments:
        return

    limited = fragments[:TOP_FRAGMENT_COUNT]
    payload = [
        (
            file_hash,
            idx + 1,
            fragment.get('substring'),
            (fragment.get('lower') or fragment.get('substring', '')).lower(),
            fragment.get('count', 0),
            fragment.get('length', len(fragment.get('substring', ''))),
        )
        for idx, fragment in enumerate(limited)
    ]
    cursor.executemany(
        '''
        INSERT INTO video_text_fragments (
            file_hash,
            rank,
            fragment_text,
            fragment_lower,
            occurrence_count,
            text_length
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''',
        payload,
    )


def _update_ocr_counts(cursor, file_hashes: List[str]):
    """Recalculate stored OCR counts for the given file hashes."""
    for file_hash in file_hashes:
        cursor.execute(
            "SELECT COUNT(*) FROM video_text WHERE file_hash = ?",
            (file_hash,),
        )
        count = cursor.fetchone()[0]
        cursor.execute(
            """
            UPDATE scanned_files
            SET ocr_text_count = ?,
                ocr_last_updated = CASE WHEN ? > 0 THEN CURRENT_TIMESTAMP ELSE ocr_last_updated END
            WHERE file_hash = ?
            """,
            (count, count, file_hash),
        )


def _recompute_fragments(cursor, file_hash: str):
    cursor.execute(
        "SELECT raw_text, occurrence_count FROM video_text WHERE file_hash = ?",
        (file_hash,),
    )
    entries = [
        (row[0], row[1])
        for row in cursor.fetchall()
        if row[0]
    ]
    fragments = calculate_top_text_fragments(entries, TOP_FRAGMENT_COUNT, MIN_OCR_TEXT_LENGTH)
    _store_fragments(cursor, file_hash, fragments)


def cleanup_ocr_text(min_length: Optional[int] = None):
    """Remove short OCR strings from the database and refresh counts."""
    threshold = min_length if min_length is not None else MIN_OCR_TEXT_LENGTH
    logger.info("Cleaning OCR entries shorter than %s characters...", threshold)

    with sqlite3.connect(DATABASE_FILE, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT file_hash FROM video_text WHERE LENGTH(TRIM(raw_text)) < ?",
            (threshold,),
        )
        affected = [row[0] for row in cursor.fetchall()]
        if not affected:
            logger.info("No OCR entries required cleanup.")
            return

        cursor.execute(
            "DELETE FROM video_text WHERE LENGTH(TRIM(raw_text)) < ?",
            (threshold,),
        )
        for file_hash in affected:
            _recompute_fragments(cursor, file_hash)
        _update_ocr_counts(cursor, affected)
        conn.commit()

    logger.info("Removed short OCR entries for %s videos.", len(affected))


def _run_refresh_sequential(jobs: List[Tuple[str, str]]) -> Tuple[int, int]:
    refreshed = 0
    skipped = 0
    for video_path, file_hash in jobs:
        result = process_video_job((video_path, file_hash))
        _, _, success, _faces, ocr_entries, ocr_fragments, error_message = result
        if not success:
            logger.warning("OCR refresh failed for %s: %s", video_path, error_message)
            skipped += 1
            continue
        if _persist_ocr_results(file_hash, ocr_entries, ocr_fragments):
            refreshed += 1
        else:
            skipped += 1
    return refreshed, skipped


def _persist_ocr_results(file_hash: str, ocr_entries, ocr_fragments) -> bool:
    try:
        with sqlite3.connect(DATABASE_FILE, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM video_text WHERE file_hash = ?', (file_hash,))
            if ocr_entries:
                payload = [
                    (
                        file_hash,
                        item.get('raw_text'),
                        item.get('normalized_text'),
                        item.get('confidence'),
                        item.get('first_seen_frame'),
                        item.get('first_seen_timestamp_ms'),
                        item.get('occurrence_count', 1),
                    )
                    for item in ocr_entries
                ]
                cursor.executemany(
                    '''
                    INSERT INTO video_text (
                        file_hash,
                        raw_text,
                        normalized_text,
                        confidence,
                        first_seen_frame,
                        first_seen_timestamp_ms,
                        occurrence_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    payload,
                )

            _store_fragments(cursor, file_hash, ocr_fragments)

            cursor.execute(
                '''
                UPDATE scanned_files
                SET ocr_text_count = ?,
                    ocr_last_updated = CURRENT_TIMESTAMP
                WHERE file_hash = ?
                ''',
                (len(ocr_entries), file_hash),
            )
            conn.commit()
        return True
    except Exception as exc:  # pragma: no cover - rare DB errors
        logger.error("Failed to persist OCR data for %s: %s", file_hash, exc)
        return False


def _process_ocr_jobs(jobs: List[Tuple[str, str]], skipped: int = 0) -> Tuple[int, int]:
    if not jobs:
        logger.info("No readable videos available for OCR processing.")
        return 0, skipped

    backend = ACTIVE_OCR_BACKEND
    worker_backend = backend
    if backend == 'easyocr':
        if pytesseract is None:
            logger.warning("EasyOCR backend cannot run in worker processes; running sequentially.")
            refreshed_seq, skipped_seq = _run_refresh_sequential(jobs)
            return refreshed_seq, skipped + skipped_seq
        logger.info("Using Tesseract backend in OCR workers to avoid EasyOCR subprocess limitations.")
        worker_backend = 'tesseract'

    num_processes = CPU_CORES_TO_USE if CPU_CORES_TO_USE is not None else cpu_count()
    num_processes = max(1, num_processes)

    refreshed = 0

    with Pool(processes=num_processes, initializer=_refresh_worker_initializer, initargs=(worker_backend,)) as pool:
        results = pool.imap_unordered(process_video_job, jobs)
        try:
            for result in results:
                file_hash, video_path, success, _faces, ocr_entries, ocr_fragments, error_message = result
                if not success:
                    logger.warning("OCR processing failed for %s: %s", video_path, error_message)
                    skipped += 1
                    continue
                if _persist_ocr_results(file_hash, ocr_entries, ocr_fragments):
                    refreshed += 1
                else:
                    skipped += 1
        finally:
            pool.close()
            pool.join()

    return refreshed, skipped


def initialize_ocr_backend():
    """Choose and initialize an OCR backend based on configuration and availability."""
    global ACTIVE_OCR_BACKEND, OCR_ENABLED, _BACKEND_INITIALIZED

    if _BACKEND_INITIALIZED:
        return

    if not OCR_ENABLED:
        ACTIVE_OCR_BACKEND = 'disabled'
        _stop_easyocr_worker()
        _BACKEND_INITIALIZED = True
        return

    if ACTIVE_OCR_BACKEND and ACTIVE_OCR_BACKEND != 'disabled':
        _BACKEND_INITIALIZED = True
        return

    logger.info("OCR engine preference: %s", OCR_ENGINE)

    preferred: List[str]
    if OCR_ENGINE == 'easyocr':
        preferred = ['easyocr', 'tesseract']
    elif OCR_ENGINE == 'tesseract':
        preferred = ['tesseract']
    else:  # auto or unknown
        preferred = ['easyocr', 'tesseract']

    for backend in preferred:
        if backend == 'easyocr':
            if easyocr is None:
                continue
            logger.info("Probing EasyOCR backend compatibility...")
            if _probe_easyocr_support():
                ACTIVE_OCR_BACKEND = 'easyocr'
                logger.info("OCR backend set to EasyOCR.")
                _BACKEND_INITIALIZED = True
                return
            logger.warning("EasyOCR backend unavailable; trying next option.")
        elif backend == 'tesseract':
            if pytesseract is None:
                logger.warning("pytesseract not installed; cannot use Tesseract backend.")
                continue
            ACTIVE_OCR_BACKEND = 'tesseract'
            _stop_easyocr_worker()
            logger.info("OCR backend set to Tesseract (pytesseract).")
            _BACKEND_INITIALIZED = True
            return

    logger.warning("No OCR backend available; disabling OCR feature.")
    OCR_ENABLED = False
    ACTIVE_OCR_BACKEND = 'disabled'
    _stop_easyocr_worker()
    _BACKEND_INITIALIZED = True


def get_ocr_reader():
    """Lazily instantiate and cache the EasyOCR reader once backend is confirmed."""
    global _ocr_reader, OCR_ENABLED, ACTIVE_OCR_BACKEND
    if not OCR_ENABLED or ACTIVE_OCR_BACKEND != 'easyocr':
        return None
    if _ocr_reader is None:
        try:
            logger.info(
                "Initializing EasyOCR reader with languages: %s (gpu=%s)",
                ",".join(config.OCR_LANGUAGES),
                config.OCR_USE_GPU,
            )
            _ocr_reader = easyocr.Reader(list(config.OCR_LANGUAGES), gpu=config.OCR_USE_GPU)
        except Exception as exc:  # pragma: no cover - hardware/env specific
            logger.error("Failed to initialize EasyOCR reader: %s", exc)
            OCR_ENABLED = False
            ACTIVE_OCR_BACKEND = 'disabled'
            return None
    return _ocr_reader


def _normalize_ocr_text(text: str) -> Optional[str]:
    """Clean whitespace and validate text length for OCR results."""
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None
    if len(cleaned) < MIN_OCR_TEXT_LENGTH or len(cleaned) > OCR_MAX_TEXT_LENGTH:
        return None
    return cleaned


def _timestamp_ms_for_frame(frame_index: int, fps: float) -> Optional[int]:
    """Convert a frame index to milliseconds using FPS when available."""
    if not fps or fps <= 0:
        return None
    return int((frame_index / fps) * 1000)


def collect_ocr_from_frame(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    aggregator: Dict[str, Dict[str, Any]],
):
    """Run OCR on a frame and merge results into the aggregator."""
    if not OCR_ENABLED:
        return

    if not _BACKEND_INITIALIZED and not mp.current_process().daemon:
        initialize_ocr_backend()
    if not _BACKEND_INITIALIZED:
        return
    if not OCR_ENABLED or ACTIVE_OCR_BACKEND in (None, 'disabled'):
        return

    timestamp_ms = _timestamp_ms_for_frame(frame_index, fps)
    backend = ACTIVE_OCR_BACKEND

    if backend == 'easyocr':
        results = _easyocr_request(frame)
        if not results:
            # Worker failed; backend may have switched to tesseract.
            if ACTIVE_OCR_BACKEND == 'tesseract':
                collect_ocr_from_frame(frame, frame_index, fps, aggregator)
            return
        for _, text, confidence in results:
            if confidence is None or confidence < OCR_MIN_CONFIDENCE:
                continue
            cleaned = _normalize_ocr_text(text)
            if not cleaned:
                continue
            _record_ocr_text(aggregator, cleaned, frame_index, timestamp_ms, confidence)

    elif backend == 'tesseract':
        if pytesseract is None:
            return
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = frame
        text = pytesseract.image_to_string(gray)
        if not text:
            return
        for line in text.splitlines():
            cleaned = _normalize_ocr_text(line)
            if not cleaned:
                continue
            _record_ocr_text(aggregator, cleaned, frame_index, timestamp_ms, None)


def serialize_ocr_entries(file_hash: str, aggregator: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare a sorted list of OCR records ready for persistence."""
    if not aggregator:
        return []
    sorted_entries = sorted(
        aggregator.values(),
        key=lambda item: (
            -(item.get("confidence") or 0.0),
            item.get("first_seen_frame", 0),
        ),
    )
    payload = []
    for item in sorted_entries:
        payload.append(
            {
                "file_hash": file_hash,
                "raw_text": item.get("raw_text"),
                "normalized_text": item.get("normalized_text"),
                "confidence": item.get("confidence"),
                "first_seen_frame": item.get("first_seen_frame"),
                "first_seen_timestamp_ms": item.get("first_seen_timestamp_ms"),
                "occurrence_count": item.get("occurrence_count", 1),
            }
        )
    return payload

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
        initialize_ocr_backend()
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
            _, _, success, faces_list, ocr_entries, ocr_fragments, error_message = result

            if success:
                face_count = len(faces_list)
                write_data_to_db(
                    faces_list,
                    [(file_hash, video_path, face_count, len(ocr_entries))],
                    None,
                    [(file_hash, ocr_entries)],
                    [(file_hash, ocr_fragments)],
                )
                successful_retries += 1
                logger.info(f"Successfully retried: {video_path}")
            else:
                write_data_to_db(
                    [],
                    [],
                    [(file_hash, video_path, error_message)],
                    [],
                    [],
                )
                logger.warning(f"Retry failed for {video_path}: {error_message}")

        logger.info(f"Retry complete. Successfully retried {successful_retries}/{len(failed_videos)} videos")

    except Exception as e:
        logger.error(f"Error during retry process: {e}")


def refresh_ocr_data(target_hashes: Optional[List[str]] = None):
    """Rebuild OCR text entries for completed videos without duplicating face data."""
    if not OCR_ENABLED:
        logger.warning("OCR is disabled. Enable OCR to refresh text data.")
        return

    initialize_ocr_backend()
    if not OCR_ENABLED or ACTIVE_OCR_BACKEND in (None, 'disabled'):
        logger.warning("No usable OCR backend; skipping OCR refresh.")
        return

    if target_hashes:
        target_hashes = list(dict.fromkeys(target_hashes))

    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        query_parts = [
            "SELECT file_hash, last_known_filepath, COALESCE(face_count, 0) AS face_count "
            "FROM scanned_files WHERE last_known_filepath IS NOT NULL"
        ]
        params: List[str] = []
        if target_hashes:
            placeholders = ','.join('?' for _ in target_hashes)
            query_parts.append(f" AND file_hash IN ({placeholders})")
            params.extend(target_hashes)
        query_parts.append(" ORDER BY (last_attempt IS NULL), last_attempt DESC, file_hash")
        rows = conn.execute(''.join(query_parts), params).fetchall()

    if not rows:
        logger.info("No videos available for OCR refresh.")
        return

    jobs = []
    skipped = 0
    for row in rows:
        file_hash = row['file_hash']
        video_path = row['last_known_filepath']
        if not video_path or not os.path.exists(video_path):
            logger.warning("Skipping OCR refresh for %s (missing file).", file_hash)
            skipped += 1
            continue
        jobs.append((video_path, file_hash))

    refreshed, skipped = _process_ocr_jobs(jobs, skipped)
    logger.info("OCR refresh complete. Updated %s videos, skipped %s.", refreshed, skipped)


def continue_ocr_data():
    """Process only videos missing OCR text entries."""
    if not OCR_ENABLED:
        logger.warning("OCR is disabled. Enable OCR to continue processing.")
        return

    initialize_ocr_backend()
    if not OCR_ENABLED or ACTIVE_OCR_BACKEND in (None, 'disabled'):
        logger.warning("No usable OCR backend; skipping OCR continue.")
        return

    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT file_hash, last_known_filepath
            FROM scanned_files
            WHERE last_known_filepath IS NOT NULL
              AND processing_status = 'completed'
              AND COALESCE(ocr_text_count, 0) = 0
            ORDER BY file_hash
            """
        ).fetchall()

    if not rows:
        logger.info("No videos pending OCR extraction.")
        return

    jobs = []
    skipped = 0
    for row in rows:
        file_hash = row['file_hash']
        video_path = row['last_known_filepath']
        if not video_path or not os.path.exists(video_path):
            logger.warning("Skipping OCR continue for %s (missing file).", file_hash)
            skipped += 1
            continue
        jobs.append((video_path, file_hash))

    refreshed, skipped = _process_ocr_jobs(jobs, skipped)
    logger.info("OCR continue complete. Updated %s videos, skipped %s.", refreshed, skipped)


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

        if 'ocr_text_count' not in columns:
            logger.info("Adding ocr_text_count column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN ocr_text_count INTEGER DEFAULT 0")

        if 'ocr_last_updated' not in columns:
            logger.info("Adding ocr_last_updated column to scanned_files table...")
            cursor.execute("ALTER TABLE scanned_files ADD COLUMN ocr_last_updated TIMESTAMP")

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
            CREATE TABLE IF NOT EXISTS video_text_fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                rank INTEGER NOT NULL,
                fragment_text TEXT NOT NULL,
                fragment_lower TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL,
                text_length INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_hash, fragment_lower),
                FOREIGN KEY (file_hash) REFERENCES scanned_files (file_hash)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_fragments_file_hash ON video_text_fragments (file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_fragments_rank ON video_text_fragments (file_hash, rank)')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                confidence REAL,
                first_seen_frame INTEGER,
                first_seen_timestamp_ms REAL,
                occurrence_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_hash) REFERENCES scanned_files (file_hash),
                UNIQUE(file_hash, normalized_text)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_text_file_hash ON video_text (file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_text_normalized ON video_text (normalized_text)')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_ocr_status ON scanned_files (ocr_text_count, processing_status)')

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

        # Metadata operations tables for smart metadata planner
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                file_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                user_note TEXT
            )
        ''')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_metadata_operations_status ON metadata_operations (status)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_metadata_operations_started ON metadata_operations (started_at DESC)'
        )

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata_operation_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                previous_comment TEXT,
                new_comment TEXT NOT NULL,
                tags_added TEXT,
                tags_removed TEXT,
                error_message TEXT,
                processed_at TIMESTAMP,
                FOREIGN KEY (operation_id) REFERENCES metadata_operations (id) ON DELETE CASCADE,
                FOREIGN KEY (file_hash) REFERENCES scanned_files (file_hash)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata_comment_cache (
                file_hash TEXT PRIMARY KEY,
                comment TEXT,
                file_mtime REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_metadata_items_operation ON metadata_operation_items (operation_id)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_metadata_items_status ON metadata_operation_items (status)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_metadata_items_file ON metadata_operation_items (file_hash)'
        )

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_item_id INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                backup_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                original_comment TEXT,
                original_metadata_json TEXT,
                FOREIGN KEY (operation_item_id) REFERENCES metadata_operation_items (id) ON DELETE CASCADE,
                FOREIGN KEY (file_hash) REFERENCES scanned_files (file_hash)
            )
        ''')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_metadata_history_file ON metadata_history (file_hash)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_metadata_history_operation ON metadata_history (operation_item_id)'
        )

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
    Returns (file_hash, video_path, success, faces_list, ocr_entries, ocr_fragments, error_message).
    """
    video_path, file_hash = job_data
    logger.info(f"Processing video: {video_path}")

    # Validate video file first
    is_valid, validation_error = validate_video_file(video_path)
    if not is_valid:
        error_msg = f"Video validation failed: {validation_error}"
        logger.error(f"Validation failed for {video_path}: {error_msg}")
        return (file_hash, video_path, False, [], [], [], error_msg)

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        faces_found_in_video = []
        ocr_aggregator: Dict[str, Dict[str, Any]] = {}
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                error_msg = "Could not open video file with OpenCV"
                logger.error(f"OpenCV error for {video_path}: {error_msg}")
                return (file_hash, video_path, False, [], [], [], error_msg)

            fps = video_capture.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = 0
            corrupted_frames = 0
            max_corrupted_frames = 10  # Allow some corrupted frames

            while True:
                try:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    if frame is None or frame.size == 0:
                        corrupted_frames += 1
                        if corrupted_frames > max_corrupted_frames:
                            error_msg = f"Too many corrupted frames ({corrupted_frames})"
                            logger.error(f"Frame corruption in {video_path}: {error_msg}")
                            video_capture.release()
                            return (file_hash, video_path, False, [], [], [], error_msg)
                        frame_count += 1
                        continue

                    if OCR_ENABLED and frame_count % OCR_FRAME_INTERVAL == 0:
                        collect_ocr_from_frame(frame, frame_count, fps, ocr_aggregator)

                    if frame_count % FRAME_SKIP == 0:
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
                    frame_count += 1

            video_capture.release()
            ocr_entries = serialize_ocr_entries(file_hash, ocr_aggregator)
            weighted_entries = [
                (item['raw_text'], item.get('occurrence_count', 1))
                for item in ocr_entries
                if item.get('raw_text')
            ]
            top_fragments = calculate_top_text_fragments(
                weighted_entries,
                TOP_FRAGMENT_COUNT,
                MIN_OCR_TEXT_LENGTH,
            )
            logger.info(
                "Successfully processed %s. Found %s faces, %s text snippets, %s top fragments.",
                video_path,
                len(faces_found_in_video),
                len(ocr_entries),
                len(top_fragments),
            )
            return (
                file_hash,
                video_path,
                True,
                faces_found_in_video,
                ocr_entries,
                top_fragments,
                None,
            )

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
    return (file_hash, video_path, False, [], [], [], error_msg)


def write_data_to_db(
    face_data,
    scanned_files_info,
    failed_files_info=None,
    ocr_text_data=None,
    ocr_fragments_data=None,
):
    """Writes collected scan results to SQLite and saves thumbnails."""
    if (
        not face_data
        and not scanned_files_info
        and not failed_files_info
        and not (ocr_text_data or [])
    ):
        return

    success_count = len(scanned_files_info) if scanned_files_info else 0
    failed_count = len(failed_files_info) if failed_files_info else 0
    ocr_count = sum(row[3] for row in scanned_files_info) if scanned_files_info else 0
    faces_count = len(face_data) if face_data else 0

    logger.info(
        "Saving progress: %s successful videos, %s failed videos, %s faces, %s OCR strings",
        success_count,
        failed_count,
        faces_count,
        ocr_count,
    )

    try:
        with sqlite3.connect(DATABASE_FILE, timeout=30) as conn:
            cursor = conn.cursor()

            # Map hashes to paths for thumbnail generation
            file_map = {h: p for h, p, _, _ in scanned_files_info} if scanned_files_info else {}
            ocr_map: Dict[str, List[Dict[str, Any]]] = (
                {file_hash: entries for file_hash, entries in ocr_text_data}
                if ocr_text_data
                else {}
            )
            fragment_map: Dict[str, List[Dict[str, Any]]] = (
                {file_hash: fragments for file_hash, fragments in ocr_fragments_data}
                if ocr_fragments_data
                else {}
            )

            # Update successful files
            if scanned_files_info:
                upsert_rows = []
                for file_hash, path, face_count, text_count in scanned_files_info:
                    manual_status = 'pending' if face_count == 0 else 'not_required'
                    upsert_rows.append(
                        (file_hash, path, 'completed', face_count, manual_status, text_count)
                    )

                cursor.executemany(
                    '''
                    INSERT INTO scanned_files (
                        file_hash,
                        last_known_filepath,
                        processing_status,
                        error_message,
                        last_attempt,
                        face_count,
                        manual_review_status,
                        ocr_text_count,
                        ocr_last_updated
                    ) VALUES (?, ?, ?, NULL, CURRENT_TIMESTAMP, ?, ?, ?, CURRENT_TIMESTAMP)
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
                        END,
                        ocr_text_count = excluded.ocr_text_count,
                        ocr_last_updated = CURRENT_TIMESTAMP
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
                        manual_review_status,
                        ocr_text_count,
                        ocr_last_updated
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, NULL, 'not_required', NULL, NULL)
                    ON CONFLICT(file_hash) DO UPDATE SET
                        last_known_filepath = excluded.last_known_filepath,
                        processing_status = excluded.processing_status,
                        error_message = excluded.error_message,
                        last_attempt = CURRENT_TIMESTAMP
                    ''',
                    [(h, p, 'failed', err) for h, p, err in failed_files_info],
                )

            # Replace OCR text entries for processed files
            processed_hashes = set()
            if ocr_map:
                for file_hash, entries in ocr_map.items():
                    processed_hashes.add(file_hash)
                    cursor.execute('DELETE FROM video_text WHERE file_hash = ?', (file_hash,))
                    if not entries:
                        continue
                    payload = [
                        (
                            file_hash,
                            item.get('raw_text'),
                            item.get('normalized_text'),
                            item.get('confidence'),
                            item.get('first_seen_frame'),
                            item.get('first_seen_timestamp_ms'),
                            item.get('occurrence_count', 1),
                        )
                        for item in entries
                    ]
                    cursor.executemany(
                        '''
                        INSERT INTO video_text (
                            file_hash,
                            raw_text,
                            normalized_text,
                            confidence,
                            first_seen_frame,
                            first_seen_timestamp_ms,
                            occurrence_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''',
                        payload,
                    )

            if fragment_map or processed_hashes:
                target_hashes = set(fragment_map.keys()) | processed_hashes
                for file_hash in target_hashes:
                    fragments = fragment_map.get(file_hash)
                    if fragments is None:
                        _recompute_fragments(cursor, file_hash)
                    else:
                        _store_fragments(cursor, file_hash, fragments)

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
        filepaths_to_hash = all_video_files
        hashed_files = {}
        total_files = len(filepaths_to_hash)
        processed_count = 0

        results_iterator = pool.imap_unordered(get_file_hash_with_path, filepaths_to_hash)

        try:
            for filepath, file_hash in results_iterator:
                if handler.shutdown_requested:
                    print("[Main] Shutdown detected during file hashing. Stopping.")
                    pool.terminate()
                    break

                processed_count += 1
                print(f"[{processed_count}/{total_files}] Hashed: {filepath} | Hash: {file_hash if file_hash else 'FAILED'}")

                if file_hash:
                    hashed_files[filepath] = file_hash
        finally:
            pool.close()
            pool.join()

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

    initialize_ocr_backend()

    num_processes = CPU_CORES_TO_USE if CPU_CORES_TO_USE is not None else cpu_count()
    print(f"Creating a pool of {num_processes} worker processes. Press Ctrl+C to stop gracefully.")

    pending_faces = []
    pending_files_info = []
    pending_failed_info = []
    pending_ocr_entries = []
    pending_ocr_fragments = []
    total_processed = 0
    total_successful = 0
    total_failed = 0

    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(process_video_job, jobs_to_process)

        try:
            for result in results_iterator:
                if handler.shutdown_requested:
                    pool.terminate()
                    break

                file_hash, video_path, success, faces_list, ocr_entries, ocr_fragments, error_message = result
                total_processed += 1

                if success:
                    total_successful += 1
                    face_count = len(faces_list)
                    ocr_unique_count = len(ocr_entries)
                    pending_files_info.append((file_hash, video_path, face_count, ocr_unique_count))
                    pending_faces.extend(faces_list)
                    pending_ocr_entries.append((file_hash, ocr_entries))
                    pending_ocr_fragments.append((file_hash, ocr_fragments))
                else:
                    total_failed += 1
                    logger.warning(f"Failed to process {video_path}: {error_message}")
                    pending_failed_info.append((file_hash, video_path, error_message))

                # Save to DB when chunk size is reached
                if len(pending_files_info) + len(pending_failed_info) >= SAVE_CHUNK_SIZE:
                    write_data_to_db(
                        pending_faces,
                        pending_files_info,
                        pending_failed_info,
                        pending_ocr_entries,
                        pending_ocr_fragments,
                    )
                    pending_faces, pending_files_info, pending_failed_info, pending_ocr_entries, pending_ocr_fragments = [], [], [], [], []
        finally:
            pool.close()
            pool.join()

    # After the loop (or on shutdown), save any remaining data
    if pending_faces or pending_files_info or pending_failed_info or pending_ocr_entries or pending_ocr_fragments:
        logger.info("Performing final save...")
        write_data_to_db(
            pending_faces,
            pending_files_info,
            pending_failed_info,
            pending_ocr_entries,
            pending_ocr_fragments,
        )

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
            _require_ffprobe()
            setup_database()
            retry_failed_videos()
            sys.exit(0)
        elif command == "cleanup":
            setup_database()
            cleanup_failed_thumbnails()
            sys.exit(0)
        elif command == "cleanup_ocr":
            setup_database()
            threshold = None
            if len(sys.argv) > 2:
                try:
                    threshold = int(sys.argv[2])
                except ValueError:
                    print("Optional threshold must be an integer.")
                    sys.exit(1)
            cleanup_ocr_text(threshold)
            sys.exit(0)
        elif command == "ocr_diagnose":
            setup_database()
            diagnose_ocr_environment()
            sys.exit(0)
        elif command == "refresh_ocr":
            setup_database()
            hashes = sys.argv[2:] if len(sys.argv) > 2 else None
            refresh_ocr_data(hashes)
            sys.exit(0)
        elif command == "continue_ocr":
            setup_database()
            continue_ocr_data()
            sys.exit(0)
        else:
            print("Usage: python scanner.py [retry|cleanup|cleanup_ocr [MIN_LEN]|refresh_ocr [FILE_HASH...]|continue_ocr|ocr_diagnose]")
            sys.exit(1)

    setup_database()
    _require_ffprobe()
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
