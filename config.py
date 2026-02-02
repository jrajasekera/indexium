from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _str_to_bool(value: str) -> bool:
    """Parse truthy environment strings into booleans."""
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Application configuration sourced from environment variables.

    Attributes:
        VIDEO_DIR: Root directory that holds source video files to scan.
        DATABASE_FILE: SQLite database path for faces, OCR, and manual-review data.
        THUMBNAIL_DIR: Directory where face thumbnails should be written.
        FRAME_SKIP: Number of frames to skip between face-detection samples while scanning.
        CPU_CORES: Maximum worker processes to use for CPU-bound scanner tasks.
        SAVE_CHUNK_SIZE: Batch size when persisting face records during scans.
        SECRET_KEY: Flask secret key enabling session cookies and flash messages.
        DEBUG: Enables Flask debug mode when true.
        DBSCAN_EPS: Face-clustering epsilon radius used by DBSCAN.
        DBSCAN_MIN_SAMPLES: Minimum neighbours required to form a DBSCAN cluster.
        FACE_DETECTION_MODEL: dlib face-detection backend ("hog" or "cnn").
        AUTO_CLASSIFY_THRESHOLD: Confidence threshold for auto-tagging suggested faces.
        NO_FACE_SAMPLE_COUNT: Number of representative frames to generate for manual review.
        NO_FACE_SAMPLE_DIR: Directory where manual-review sample frames are stored.
        MANUAL_VIDEO_REVIEW_ENABLED: Master switch for the manual-review UI and routes.
        MANUAL_NAME_SUGGEST_THRESHOLD: Minimum fuzzy-match score to auto-suggest a name.
        MANUAL_REVIEW_WARMUP_ENABLED: Enables background warmup of the next manual-review video.
        MANUAL_REVIEW_WARMUP_WORKERS: Size of the thread pool used for warmup jobs.
        MANUAL_REVIEW_WARMUP_DEPTH: Number of upcoming manual-review videos to prewarm.
        MANUAL_KNOWN_PEOPLE_CACHE_SECONDS: TTL for caching the list of known people names.
        OCR_ENABLED: Enables OCR extraction during scanning when true.
        OCR_ENGINE: Preferred OCR engine identifier.
        OCR_LANGUAGES: Tuple of language codes to pass to the OCR engine.
        OCR_USE_GPU: Enables GPU acceleration for OCR, if supported.
        OCR_FRAME_INTERVAL: Frame interval the OCR engine should sample.
        OCR_MIN_CONFIDENCE: Minimum OCR confidence to persist text results.
        OCR_MIN_TEXT_LENGTH: Minimum number of characters needed to keep OCR text.
        OCR_MAX_TEXT_LENGTH: Maximum allowed OCR text snippet length.
        OCR_MAX_RESULTS_PER_VIDEO: Cap on total OCR text entries per video.
        OCR_TOP_FRAGMENT_COUNT: Number of top text fragments to compute per video.
    """

    VIDEO_DIR: str = os.environ.get("INDEXIUM_VIDEO_DIR", "test_videos")
    DATABASE_FILE: str = os.environ.get("INDEXIUM_DB", "video_faces.db")
    THUMBNAIL_DIR: str = "thumbnails"
    FRAME_SKIP: int = int(os.environ.get("FRAME_SKIP", "25"))

    _cpu = os.environ.get("CPU_CORES", "6")
    CPU_CORES: int | None = None if _cpu is None or _cpu.lower() == "none" else int(_cpu)

    SAVE_CHUNK_SIZE: int = int(os.environ.get("SAVE_CHUNK_SIZE", "4"))

    SECRET_KEY: str = os.environ.get("SECRET_KEY", os.urandom(24).hex())
    DEBUG: bool = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    DBSCAN_EPS: float = float(os.environ.get("DBSCAN_EPS", "0.4"))
    DBSCAN_MIN_SAMPLES: int = int(os.environ.get("DBSCAN_MIN_SAMPLES", "5"))
    FACE_DETECTION_MODEL: str = os.environ.get("FACE_DETECTION_MODEL", "hog")
    AUTO_CLASSIFY_THRESHOLD: float = float(os.environ.get("AUTO_CLASSIFY_THRESHOLD", "0.3"))

    NO_FACE_SAMPLE_COUNT: int = int(os.environ.get("NO_FACE_SAMPLE_COUNT", "25"))
    NO_FACE_SAMPLE_DIR: str = os.environ.get(
        "NO_FACE_SAMPLE_DIR",
        os.path.join(THUMBNAIL_DIR, "no_faces"),
    )
    MANUAL_VIDEO_REVIEW_ENABLED: bool = _str_to_bool(
        os.environ.get("MANUAL_VIDEO_REVIEW_ENABLED", "true")
    )
    MANUAL_NAME_SUGGEST_THRESHOLD: float = float(
        os.environ.get("MANUAL_NAME_SUGGEST_THRESHOLD", "0.82")
    )
    MANUAL_REVIEW_WARMUP_ENABLED: bool = _str_to_bool(
        os.environ.get("MANUAL_REVIEW_WARMUP_ENABLED", "true")
    )
    MANUAL_REVIEW_WARMUP_WORKERS: int = int(os.environ.get("MANUAL_REVIEW_WARMUP_WORKERS", "4"))
    MANUAL_REVIEW_WARMUP_DEPTH: int = int(os.environ.get("MANUAL_REVIEW_WARMUP_DEPTH", "2"))
    MANUAL_KNOWN_PEOPLE_CACHE_SECONDS: float = float(
        os.environ.get("MANUAL_KNOWN_PEOPLE_CACHE_SECONDS", "30")
    )

    METADATA_PLAN_WORKERS: int = max(
        1,
        int(os.environ.get("METADATA_PLAN_WORKERS", "8")),
    )

    OCR_ENABLED: bool = _str_to_bool(os.environ.get("INDEXIUM_OCR_ENABLED", "true"))
    OCR_ENGINE: str = os.environ.get("INDEXIUM_OCR_ENGINE", "auto")
    _ocr_langs = os.environ.get("INDEXIUM_OCR_LANGS", "en")
    OCR_LANGUAGES: tuple[str, ...] = tuple(
        part.strip() for part in _ocr_langs.split(",") if part.strip()
    ) or ("en",)
    OCR_USE_GPU: bool = _str_to_bool(os.environ.get("INDEXIUM_OCR_USE_GPU", "false"))
    OCR_FRAME_INTERVAL: int = int(os.environ.get("INDEXIUM_OCR_FRAME_INTERVAL", "60"))
    OCR_MIN_CONFIDENCE: float = float(os.environ.get("INDEXIUM_OCR_MIN_CONFIDENCE", "0.5"))
    OCR_MIN_TEXT_LENGTH: int = int(os.environ.get("INDEXIUM_OCR_MIN_TEXT_LENGTH", "3"))
    OCR_MAX_TEXT_LENGTH: int = int(os.environ.get("INDEXIUM_OCR_MAX_TEXT_LENGTH", "80"))
    OCR_MAX_RESULTS_PER_VIDEO: int = int(os.environ.get("INDEXIUM_OCR_MAX_RESULTS", "200"))
    OCR_TOP_FRAGMENT_COUNT: int = int(os.environ.get("INDEXIUM_OCR_TOP_FRAGMENTS", "10"))

    # NFO Metadata Settings
    NFO_REMOVE_STALE_ACTORS: bool = _str_to_bool(os.environ.get("NFO_REMOVE_STALE_ACTORS", "true"))
    NFO_BACKUP_MAX_AGE_DAYS: int = int(os.environ.get("NFO_BACKUP_MAX_AGE_DAYS", "30"))
