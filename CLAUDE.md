# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indexium is a Python-based video face scanning and tagging application. It detects faces in videos, clusters them by person using ML, and provides a web UI for tagging and metadata management.

## Commands

### Development
```bash
uv sync                          # Install dependencies (preferred)
pip install -e .                 # Alternative installation
python app.py                    # Start web UI at http://localhost:5001
INDEXIUM_VIDEO_DIR=/path python scanner.py  # Run face scanner
```

### Testing
```bash
pytest -q                        # Run all tests
pytest -q tests/test_scanner.py::test_cluster_faces_updates_ids  # Single test
python e2e_test.py test_vids     # End-to-end pipeline test (uses temp DB)
```

### Scanner Commands
```bash
python scanner.py                        # Full scan
python scanner.py refresh_ocr            # Refresh OCR for all completed videos
python scanner.py refresh_ocr HASH123    # Refresh specific file hash
python scanner.py continue_ocr           # Process videos missing OCR
python scanner.py cleanup_ocr            # Remove short OCR text (default <4 chars)
python scanner.py cleanup_ocr 6          # Custom minimum length
```

## Architecture

### Core Modules

- **app.py**: Flask web application with routes for tagging UI, face management, metadata operations, and manual video review workflow. Uses SQLite connection per-request pattern with `get_db_connection()`/`close_db_connection()`.

- **scanner.py**: Video processing pipeline - face detection via `face_recognition` library, DBSCAN clustering, OCR extraction (EasyOCR with Tesseract fallback), multiprocessing workers. Entry point for CLI scanner commands.

- **metadata_services.py**: Metadata planning and writing system with these key classes:
  - `MetadataPlanner`: Generates plans comparing DB tags vs file metadata
  - `MetadataWriter`: Executes write operations with pause/resume/cancel support
  - `BackupManager`: Creates/restores file backups before metadata writes
  - `HistoryService`: Tracks operation history for rollback capability

- **config.py**: Centralized configuration via `Config` class. All settings loaded from environment variables with defaults. Always modify settings here, not inline.

- **text_utils.py**: OCR text fragment ranking/filtering via `calculate_top_text_fragments()`.

- **util.py**: File hashing for content-based video tracking.

### Database Schema (SQLite)

- `scanned_files`: Tracks processed videos by content hash, stores face counts, manual review status, sampling seeds
- `faces`: Face data with locations, 128-dim encodings, cluster IDs, and person tags
- `video_people`: Links videos-without-faces to manually assigned person tags
- `video_text`: OCR text snippets keyed by video hash

### Processing Pipeline

1. Scanner processes videos frame-by-frame extracting face encodings (128-dim vectors)
2. DBSCAN clusters similar encodings (eps=0.4, min_samples=5)
3. Web UI presents clusters for manual naming
4. Metadata writer embeds "People: Name1, Name2" into video file comments via ffmpeg

### Key Patterns

- **File tracking**: SHA256 hash of first+last 25 blocks identifies videos regardless of rename/move
- **Graceful shutdown**: `SignalHandler` class catches Ctrl+C and saves progress
- **OCR fallback**: EasyOCR preferred, falls back to Tesseract if unavailable
- **Caching**: Known people cache in app.py invalidated on tag changes

## Configuration

Key environment variables (see `config.py` for full list):
- `INDEXIUM_VIDEO_DIR`: Video source directory
- `INDEXIUM_DB`: SQLite database path (default: `video_faces.db`)
- `FRAME_SKIP`: Frames between face detection samples (default: 25)
- `CPU_CORES`: Worker processes (default: all cores)
- `INDEXIUM_OCR_ENABLED`: Toggle OCR (default: true)
- `INDEXIUM_OCR_ENGINE`: `easyocr`, `tesseract`, or `auto`

## Testing Notes

- Tests use monkeypatching to redirect DB/paths to temp locations
- Check `tests/conftest.py` for shared fixtures
- E2E test (`e2e_test.py`) runs full pipeline with isolated temp directory
