# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indexium is a Python-based video face scanning and tagging application. It detects faces in videos, clusters them by person using ML, and provides a web UI for tagging and metadata management. It also extracts on-screen text via OCR and supports manual video tagging for videos without detected faces.

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
pytest --cov --cov-report=term-missing  # Run tests with coverage
pytest --cov --cov-report=html          # Generate HTML report in htmlcov/
```

### Scanner Commands
```bash
python scanner.py                        # Full scan
python scanner.py retry                  # Retry processing of failed videos
python scanner.py cleanup                # Clean up orphaned thumbnail files
python scanner.py refresh_ocr            # Refresh OCR for all completed videos
python scanner.py refresh_ocr HASH123    # Refresh specific file hash
python scanner.py continue_ocr           # Process videos missing OCR
python scanner.py cleanup_ocr            # Remove short OCR text (default <4 chars)
python scanner.py cleanup_ocr 6          # Custom minimum length
python scanner.py ocr_diagnose           # Diagnose OCR environment setup
```

## Architecture

### Core Modules

- **app.py**: Flask web application with routes for tagging UI, face management, metadata operations, and manual video review workflow. Uses SQLite connection per-request pattern with `get_db_connection()`/`close_db_connection()`. Includes known-people caching and background warmup for manual review.

- **scanner.py**: Video processing pipeline - face detection via `face_recognition` library, DBSCAN clustering, OCR extraction (EasyOCR with Tesseract fallback), multiprocessing workers. Entry point for CLI scanner commands. Requires ffprobe at startup.

- **metadata_services.py**: Metadata planning and writing system with these key classes:
  - `MetadataPlanner`: Generates plans comparing DB tags vs file metadata
  - `MetadataWriter`: Executes write operations with pause/resume/cancel support
  - `BackupManager`: Creates/restores file backups before metadata writes
  - `HistoryService`: Tracks operation history for rollback capability

- **nfo_services.py**: NFO metadata file management for Jellyfin integration with these key classes:
  - `NfoService`: Reads/writes actor lists in NFO XML files, preserving non-indexium actors
  - `NfoPlanner`: Builds change plans comparing DB people vs existing NFO actors, with risk levels (safe/warning/danger/blocked)
  - `NfoWriter`: Asynchronous background writer with pause/resume/cancel support
  - `NfoBackupManager`: Operation-scoped NFO file backup/restore
  - `NfoHistoryService`: Query and rollback NFO operations

- **config.py**: Centralized configuration via `Config` dataclass. All settings loaded from environment variables with defaults. Always modify settings here, not inline.

- **type_defs.py**: Centralized type definitions using `TypedDict` and numpy aliases. Covers DB row types, API response types, face/OCR data structures, cache structures, and processing types.

- **text_utils.py**: OCR text fragment ranking/filtering via `calculate_top_text_fragments()`.

- **util.py**: File hashing (`get_file_hash()`) for content-based video tracking.

- **signal_handler.py**: `SignalHandler` class for graceful shutdown on Ctrl+C.

- **e2e_test.py**: End-to-end test runner with `run_pipeline()` and `main()` functions.

### Database Schema (SQLite)

- `scanned_files`: Tracks processed videos by content hash, stores face counts, manual review status, sampling seeds, OCR text counts
- `faces`: Face data with locations, 128-dim encodings, cluster IDs, and person tags (includes auto-suggestion columns)
- `video_people`: Links videos-without-faces to manually assigned person tags
- `video_text`: OCR text snippets keyed by video hash (raw_text, normalized_text, confidence, timestamps)
- `video_text_fragments`: Top-ranked OCR text fragments per video for UI display
- `metadata_operations`: Tracks metadata write operations (status, counts, timestamps)
- `metadata_operation_items`: Individual file items within a metadata operation (per-file status, old/new comments, NFO path)
- `metadata_history`: Backup of original metadata before writes for rollback
- `metadata_comment_cache`: Caches video file comments keyed by file hash with mtime tracking
- `nfo_actor_cache`: Caches parsed NFO file actor data keyed by NFO path with mtime tracking

### Processing Pipeline

1. Scanner processes videos frame-by-frame extracting face encodings (128-dim vectors)
2. DBSCAN clusters similar encodings (eps=0.4, min_samples=5)
3. OCR extracts on-screen text during scanning (configurable interval/confidence)
4. Web UI presents clusters for manual naming
5. Manual video review workflow for videos without detected faces
6. Metadata writer embeds "People: Name1, Name2" into video file comments via ffmpeg
7. NFO writer updates Jellyfin-compatible NFO files with actor metadata

### Key Patterns

- **File tracking**: SHA256 hash of first+last 25 blocks identifies videos regardless of rename/move
- **Graceful shutdown**: `SignalHandler` class catches Ctrl+C and saves progress
- **OCR fallback**: EasyOCR preferred (via subprocess worker), falls back to Tesseract if unavailable
- **Caching**: Known people cache in app.py with TTL, invalidated on tag changes
- **Background warmup**: Pre-generates sample frames for next manual review videos (configurable depth and worker count)
- **NFO integration**: Reads/writes Jellyfin NFO XML files, preserving non-indexium actors, with risk-based planning and rollback

## Configuration

Key environment variables (see `config.py` for full list):

### Core Settings
- `INDEXIUM_VIDEO_DIR`: Video source directory (default: `test_videos`)
- `INDEXIUM_DB`: SQLite database path (default: `video_faces.db`)
- `FRAME_SKIP`: Frames between face detection samples (default: 25)
- `CPU_CORES`: Worker processes (default: 6, use `none` for all cores)
- `SAVE_CHUNK_SIZE`: Batch size for DB writes (default: 4)

### Face Detection
- `DBSCAN_EPS`: Clustering epsilon radius (default: 0.4)
- `DBSCAN_MIN_SAMPLES`: Minimum cluster size (default: 5)
- `FACE_DETECTION_MODEL`: `hog` or `cnn` (default: `hog`)
- `AUTO_CLASSIFY_THRESHOLD`: Auto-tagging confidence threshold (default: 0.3)

### Manual Video Review
- `MANUAL_VIDEO_REVIEW_ENABLED`: Toggle workflow (default: true)
- `NO_FACE_SAMPLE_COUNT`: Frames per manual review video (default: 25)
- `NO_FACE_SAMPLE_DIR`: Sample frame storage (default: `thumbnails/no_faces`)
- `MANUAL_NAME_SUGGEST_THRESHOLD`: Fuzzy match threshold (default: 0.82)
- `MANUAL_REVIEW_WARMUP_ENABLED`: Background warmup (default: true)
- `MANUAL_REVIEW_WARMUP_WORKERS`: Thread pool size for warmup jobs (default: 4)
- `MANUAL_REVIEW_WARMUP_DEPTH`: Number of upcoming videos to prewarm (default: 10)
- `MANUAL_KNOWN_PEOPLE_CACHE_SECONDS`: Cache TTL (default: 30)

### OCR Settings
- `INDEXIUM_OCR_ENABLED`: Toggle OCR (default: true)
- `INDEXIUM_OCR_ENGINE`: `easyocr`, `tesseract`, or `auto` (default: auto)
- `INDEXIUM_OCR_LANGS`: Comma-separated language codes (default: `en`)
- `INDEXIUM_OCR_USE_GPU`: GPU acceleration (default: false)
- `INDEXIUM_OCR_FRAME_INTERVAL`: Frames between OCR samples (default: 60)
- `INDEXIUM_OCR_MIN_CONFIDENCE`: Minimum confidence (default: 0.5)
- `INDEXIUM_OCR_MIN_TEXT_LENGTH`: Minimum text length (default: 3)
- `INDEXIUM_OCR_MAX_TEXT_LENGTH`: Maximum text length (default: 80)
- `INDEXIUM_OCR_MAX_RESULTS`: Max OCR entries per video (default: 200)
- `INDEXIUM_OCR_TOP_FRAGMENTS`: Top fragments to save (default: 10)

### NFO Settings
- `NFO_REMOVE_STALE_ACTORS`: Remove stale actors from NFO files (default: true)
- `NFO_BACKUP_MAX_AGE_DAYS`: Max age for NFO backup files (default: 30)

### Flask/Metadata
- `SECRET_KEY`: Flask secret key
- `FLASK_DEBUG`: Debug mode (default: false)
- `METADATA_PLAN_WORKERS`: Parallel workers for metadata planning (default: 8)

## Testing Notes

- Tests use monkeypatching to redirect DB/paths to temp locations
- Check `tests/conftest.py` for shared fixtures
- E2E test (`e2e_test.py`) runs full pipeline with isolated temp directory
- Test files: `test_app.py`, `test_scanner.py`, `test_metadata_services.py`, `test_metadata_writer.py`, `test_nfo_services.py`, `test_config.py`, `test_util.py`, `test_text_utils.py`, `test_signal_handler.py`, `test_e2e.py`, `test_e2e_ui.py`

## Manual UI Testing with Playwright

To manually verify UI changes, use the Playwright MCP tools to interact with the running app.

### 1. Check if the app is running
```bash
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5001/ || echo "not running"
```

### 2. Start the app in background (if not running)
```bash
python app.py &
```
Note: Track whether you started the app so you know to shut it down later.

### 3. Navigate and interact using Playwright MCP tools
- `browser_navigate` - Navigate to a URL (e.g., `http://127.0.0.1:5001/videos/manual`)
- `browser_snapshot` - Capture accessibility snapshot to see page structure and element refs
- `browser_take_screenshot` - Take a visual screenshot to verify layout/styling
- `browser_click` - Click elements using refs from snapshot
- `browser_type` - Type into input fields
- `browser_fill_form` - Fill multiple form fields at once

### 4. Shut down the app (if you started it)
```bash
pkill -f "python app.py"
```
Only kill the app if you were the one who started it in step 2.
