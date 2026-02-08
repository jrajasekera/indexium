"""Type definitions for the Indexium codebase."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sqlite3 import Row

# =============================================================================
# Numpy type aliases
# =============================================================================

# 128-dimensional face encoding vector
FaceEncoding = npt.NDArray[np.float64]

# BGR/RGB video frame (height x width x 3)
VideoFrame = npt.NDArray[np.uint8]


# =============================================================================
# Database Row TypedDicts
# =============================================================================


class ScannedFileRow(TypedDict):
    """Row from scanned_files table."""

    file_hash: str
    last_known_filepath: str


class FaceRow(TypedDict):
    """Row from faces table."""

    id: int
    file_hash: str
    frame_number: int
    face_location: str  # JSON string of (top, right, bottom, left)
    face_encoding: bytes  # Pickled numpy array
    cluster_id: int | None
    person_name: str | None


class FaceRowPartial(TypedDict, total=False):
    """Partial row from faces table (for queries with subset of columns)."""

    id: int
    file_hash: str
    frame_number: int
    face_location: str
    face_encoding: bytes
    cluster_id: int
    person_name: str


class VideoTextRow(TypedDict):
    """Row from video_text table."""

    id: int
    file_hash: str
    raw_text: str
    normalized_text: str
    confidence: float
    first_frame: int
    last_frame: int
    occurrence_count: int
    timestamp_seconds: float


class VideoTextFragmentRow(TypedDict):
    """Row from video_text_fragments table."""

    id: int
    file_hash: str
    fragment_text: str
    occurrence_count: int
    total_length: int
    rank: int


class VideoPeopleRow(TypedDict):
    """Row from video_people table."""

    id: int
    file_hash: str
    person_name: str


# =============================================================================
# Cache structures
# =============================================================================


class KnownPeopleCache(TypedDict):
    """Cache for known people data in app.py."""

    timestamp: float
    names: list[str]
    prepared: list[PreparedPerson]


class PreparedPerson(TypedDict):
    """Pre-processed person data for fuzzy matching."""

    original: str
    normalized: str


# =============================================================================
# OCR types
# =============================================================================


class OCRAggregatorEntry(TypedDict):
    """Entry in OCR aggregator dict."""

    normalized: str
    raw: str
    confidence: float
    first_frame: int
    last_frame: int
    count: int


class OCREntry(TypedDict):
    """Serialized OCR entry for database storage."""

    file_hash: str
    raw_text: str
    normalized_text: str
    confidence: float
    first_frame: int
    last_frame: int
    occurrence_count: int
    timestamp_seconds: float


class TextFragment(TypedDict):
    """Top OCR text fragment result."""

    fragment_text: str
    occurrence_count: int
    total_length: int


# =============================================================================
# Sample/Thumbnail types
# =============================================================================


class SampleInfo(TypedDict):
    """Information about a sample frame for manual review."""

    path: str
    frame_number: int
    timestamp: str


# =============================================================================
# API response types
# =============================================================================


class VideoInfo(TypedDict, total=False):
    """Video information returned by API endpoints."""

    file_hash: str
    filepath: str
    face_count: int
    faces: list[FaceInfo]
    ocr_text: list[OCRTextInfo]
    text_fragments: list[TextFragmentInfo]
    people: list[str]
    samples: list[SampleInfo]
    needs_review: bool
    manual_reviewed: int
    file_exists: bool


class FaceInfo(TypedDict, total=False):
    """Face information for API responses."""

    id: int
    frame_number: int
    cluster_id: int | None
    person_name: str | None
    thumbnail_path: str | None


class OCRTextInfo(TypedDict):
    """OCR text information for API responses."""

    raw_text: str
    normalized_text: str
    confidence: float
    timestamp_seconds: float
    occurrence_count: int


class TextFragmentInfo(TypedDict):
    """Text fragment info for API responses."""

    fragment_text: str
    occurrence_count: int
    total_length: int
    rank: int


# =============================================================================
# Processing types
# =============================================================================


class FaceData(TypedDict):
    """Face data collected during video processing."""

    file_hash: str
    frame_number: int
    face_location: tuple[int, int, int, int]  # (top, right, bottom, left)
    face_encoding: FaceEncoding


class ClusterResult(TypedDict):
    """Result from face clustering."""

    cluster_id: int
    face_ids: list[int]
    representative_encoding: FaceEncoding | None


# =============================================================================
# Metadata types
# =============================================================================


class MetadataWriteResult(TypedDict):
    """Result from metadata write operation."""

    success: bool
    file_path: str
    error: str | None


class PlanItemDict(TypedDict, total=False):
    """Serialized plan item for API responses."""

    file_hash: str
    file_path: str
    db_people: list[str]
    existing_comment: str | None
    result_comment: str
    file_missing: bool
    requires_update: bool
    will_overwrite_custom: bool
    category: str
    file_modified_time: float | None


# =============================================================================
# Type guards and utilities
# =============================================================================


def row_to_dict(row: Row) -> dict[str, object]:
    """Convert an sqlite3.Row to a dictionary."""
    return dict(zip(row.keys(), row, strict=False))
