import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration loaded from environment variables."""
    VIDEO_DIR: str = os.environ.get("INDEXIUM_VIDEO_DIR", "test_videos")
    DATABASE_FILE: str = os.environ.get("INDEXIUM_DB", "video_faces.db")
    THUMBNAIL_DIR: str = "thumbnails"
    FRAME_SKIP: int = int(os.environ.get("FRAME_SKIP", "25"))

    _cpu = os.environ.get("CPU_CORES")
    CPU_CORES: Optional[int] = None if _cpu is None or _cpu.lower() == "none" else int(_cpu)

    SAVE_CHUNK_SIZE: int = int(os.environ.get("SAVE_CHUNK_SIZE", "10"))

    SECRET_KEY: str = os.environ.get("SECRET_KEY", os.urandom(24).hex())
    DEBUG: bool = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    DBSCAN_EPS: float = float(os.environ.get("DBSCAN_EPS", "0.4"))
    DBSCAN_MIN_SAMPLES: int = int(os.environ.get("DBSCAN_MIN_SAMPLES", "5"))
    FACE_DETECTION_MODEL: str = os.environ.get("FACE_DETECTION_MODEL", "hog")
    AUTO_CLASSIFY_THRESHOLD: float = float(os.environ.get("AUTO_CLASSIFY_THRESHOLD", "0.3"))
