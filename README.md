# Indexium

A Python-based video face scanning and tagging application that automatically detects faces in videos, groups them by person using machine learning clustering, and allows you to manually tag and organize people found in your video collection.

## Features

- **Automatic Face Detection**: Scans video files and extracts faces using computer vision (HOG or CNN models)
- **Smart Clustering**: Groups similar faces together using DBSCAN clustering algorithm
- **Web-based Tagging Interface**: Clean, responsive web UI for reviewing and naming face groups
- **Parallel Processing**: Multi-core video processing for faster scanning
- **File Hash-based Tracking**: Tracks videos by content hash, handles moved/renamed files gracefully
- **Smart Metadata Planner**: Builds a risk-scored plan before writing metadata changes
- **NFO/Jellyfin Integration**: Reads and writes Jellyfin-compatible NFO actor metadata with backup and rollback support
- **On-Screen Text Capture**: Extracts unique OCR snippets from each video and surfaces them during tagging for quick copy/paste workflows
- **Face Group Management**: Split groups, merge people, rename, and organize your tags
- **Remove False Positives**: Delete mistaken face detections directly from the web UI
- **Progress Tracking**: Visual progress indicators and statistics
- **Manual Video Tagging Workflow**: Review videos without detected faces using sampled frames and assign people tags
- **Background Warmup**: Pre-generates sample frames for upcoming manual-review videos to improve responsiveness

## Screenshots

The application provides an intuitive web interface for:
- Reviewing untagged face groups
- Naming individuals found in videos
- Managing and editing existing tags
- Viewing all identified people in your collection

## Installation

### Prerequisites

- Python 3.10 or higher
- ffmpeg/ffprobe available on `PATH`
- OpenCV dependencies (for video processing)
- dlib dependencies (for face recognition)
- EasyOCR requirements (PyTorch + torchvision); installed automatically via pip/uv, but ensure CUDA/cuDNN packages are available if you plan to use GPU OCR

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd indexium
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### System Dependencies

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev python3-dev ffmpeg
# For OCR fallback
sudo apt install tesseract-ocr
```

On macOS:
```bash
brew install cmake ffmpeg
# For OCR fallback
brew install tesseract
```

## Usage

### 1. Set Environment Variable

Set the path to your video directory:
```bash
export INDEXIUM_VIDEO_DIR="/path/to/your/videos"
```

### 2. Scan Videos

Run the scanner to detect and cluster faces:
```bash
python scanner.py
```

The scanner will:
- Process all video files in your directory (MP4, MKV, MOV, AVI)
- Extract faces from video frames
- Group similar faces using machine learning
- Save results to a SQLite database
- Capture on-screen text snippets (when OCR is enabled) and store them alongside face data for later tagging
  - If EasyOCR cannot run on your hardware, Indexium automatically falls back to Tesseract (requires the `tesseract` binary)

### 3. Tag Faces

Start the web application:
```bash
python app.py
```

Then open your browser to `http://localhost:5001` to:
- Review face groups and assign names
- Split incorrectly grouped faces
- Merge faces of the same person
- Manage your tagged people

### 4. Tag Videos Without Faces

Open `/videos/manual` in the web UI when a video has no detected faces. Indexium samples a grid of frames so you can tag anyone who appears or mark the clip as containing no recognizable people. Use the "Reshuffle" button for a new random set of frames, and mark videos as done (or no people) once reviewed.

### 5. Write NFO Metadata

Once you've tagged people, use the Smart Metadata Planner (`/metadata_preview`) to apply NFO updates:
- Review risk levels (`safe`/`warning`/`danger`/`blocked`) before writing
- Click **Write Selected Metadata** to start an async write operation
- Indexium updates `<actor source="indexium">` entries and preserves non-indexium actors
- If no NFO exists yet, Indexium writes a new default `<video_name>.nfo` beside the video
- The writer supports pause, resume, cancel, and rollback from metadata history

Note: the default web workflow writes NFO metadata (Jellyfin style), not ffmpeg comment tags inside video files.

### 6. Refresh OCR Text (optional)

If you tweak OCR settings or install new language packs, rebuild stored text snippets without touching face data:

```bash
python scanner.py refresh_ocr            # refresh all completed videos
python scanner.py refresh_ocr HASH123…   # refresh specific file hashes
```

If you change filtering rules or want to purge short snippets, run the cleanup pass:

```bash
python scanner.py cleanup_ocr           # drops OCR text shorter than 4 chars (current default)
python scanner.py cleanup_ocr 6         # optional custom minimum length
```

`refresh_ocr` now runs in parallel, reusing the same worker pool logic as a full scan. If EasyOCR is configured but unavailable in worker processes, the command automatically falls back to Tesseract (sequentially if no GPU-safe backend exists).

To process only videos that are missing OCR text, use:

```bash
python scanner.py continue_ocr
```

### 7. Other Scanner Commands

```bash
python scanner.py retry          # Retry processing of previously failed videos
python scanner.py cleanup        # Clean up orphaned thumbnail files
python scanner.py ocr_diagnose   # Diagnose OCR environment setup
```

## Configuration

All configuration is centralized in `config.py`. Values are loaded from
environment variables with sensible defaults:

### Core Settings
- `INDEXIUM_VIDEO_DIR`: directory of videos to scan (default: `test_videos`)
- `INDEXIUM_DB`: path to the SQLite database (default: `video_faces.db`)
- `FRAME_SKIP`: how many frames to skip between scans (default: 25)
- `CPU_CORES`: number of CPU cores to use (default: 6, `none` for all cores)
- `SAVE_CHUNK_SIZE`: how often to save progress (default: 4)

### Face Detection & Clustering
- `FACE_DETECTION_MODEL`: `hog` or `cnn` (default: `hog`)
- `DBSCAN_EPS`: clustering epsilon radius (default: 0.4)
- `DBSCAN_MIN_SAMPLES`: minimum cluster size (default: 5)
- `AUTO_CLASSIFY_THRESHOLD`: distance threshold for automatic face naming (default: 0.3)

### Manual Video Review
- `MANUAL_VIDEO_REVIEW_ENABLED`: toggle the manual video tagging workflow (default: `true`)
- `NO_FACE_SAMPLE_COUNT`: number of frames to sample per manual-review video (default: 25)
- `NO_FACE_SAMPLE_DIR`: directory for cached manual-review frame images (default: `thumbnails/no_faces`)
- `MANUAL_NAME_SUGGEST_THRESHOLD`: fuzzy match threshold for name suggestions (default: 0.82)
- `MANUAL_REVIEW_WARMUP_ENABLED`: enable background warmup for upcoming manual-review videos (default: `true`)
- `MANUAL_REVIEW_WARMUP_WORKERS`: number of warmup threads (default: 4)
- `MANUAL_REVIEW_WARMUP_DEPTH`: how many upcoming videos to prewarm (default: 10)
- `MANUAL_KNOWN_PEOPLE_CACHE_SECONDS`: cache TTL for known people list (default: 30)

### OCR Settings
- `INDEXIUM_OCR_ENABLED`: toggle OCR extraction during scanning (default: `true`)
- `INDEXIUM_OCR_ENGINE`: choose `easyocr`, `tesseract`, or `auto` (default; tries EasyOCR then falls back to Tesseract)
- `INDEXIUM_OCR_LANGS`: comma-separated EasyOCR language codes (default: `en`)
- `INDEXIUM_OCR_USE_GPU`: enable GPU acceleration for OCR (default: `false`)
- `INDEXIUM_OCR_FRAME_INTERVAL`: frames to skip between OCR samples (default: 60)
- `INDEXIUM_OCR_MIN_CONFIDENCE`: minimum EasyOCR confidence (0.0–1.0, default: 0.5)
- `INDEXIUM_OCR_MIN_TEXT_LENGTH`: minimum text length to keep (default: 3)
- `INDEXIUM_OCR_MAX_TEXT_LENGTH`: maximum text length retained (default: 80)
- `INDEXIUM_OCR_MAX_RESULTS`: cap of unique OCR strings stored per video (default: 200)
- `INDEXIUM_OCR_TOP_FRAGMENTS`: number of highest-ranked substrings saved per video (default: 10)

### Metadata & NFO
- `SECRET_KEY`: Flask secret key
- `FLASK_DEBUG`: run the web UI in debug mode (default: `false`)
- `METADATA_PLAN_WORKERS`: worker count for the legacy ffmpeg-comment planner (`metadata_services.py`) (default: 8)
- `NFO_REMOVE_STALE_ACTORS`: reserved config for NFO actor policy (currently not applied by the active writer)
- `NFO_BACKUP_MAX_AGE_DAYS`: reserved config for NFO backup retention tooling (backup cleanup is not automatic yet)

## Database

The application uses SQLite with these main tables:
- `scanned_files`: Tracks processed videos by hash, face counts, manual review status, and cached sampling seeds
- `faces`: Stores face data, locations, 128-dim encodings, cluster IDs, and person tags (includes auto-suggestion columns)
- `video_people`: Manual tags that link videos-without-faces to the people who appear in them
- `video_text`: OCR-derived text snippets keyed by video hash, with confidence and occurrence stats
- `video_text_fragments`: Top-ranked OCR text fragments per video for UI display
- `metadata_operations`: Tracks metadata write operations (status, counts, timestamps)
- `metadata_operation_items`: Individual file items within a metadata operation (per-file status, old/new comments, NFO path)
- `metadata_history`: Backup records used for metadata rollback
- `metadata_comment_cache`: Legacy cache table used by the ffmpeg-comment metadata services
- `nfo_actor_cache`: Reserved table for NFO actor caching (schema in place)

Database file: `video_faces.db`

## File Structure

```
indexium/
├── app.py                # Flask web application and API routes
├── scanner.py            # Video scanning, face clustering, and OCR pipeline
├── config.py             # Centralized configuration via environment variables
├── metadata_services.py  # Legacy ffmpeg-comment metadata services (compatibility/tests)
├── nfo_services.py       # NFO/Jellyfin metadata file management
├── type_defs.py          # Centralized TypedDict and type definitions
├── text_utils.py         # OCR text fragment ranking and filtering
├── util.py               # File hashing utility
├── signal_handler.py     # Graceful shutdown handler
├── e2e_test.py           # End-to-end pipeline test runner
├── templates/            # Flask HTML templates
├── tests/                # Pytest test suite
├── pyproject.toml        # Project dependencies
├── video_faces.db        # SQLite database (created on first run)
└── README.md             # This file
```

## How It Works

1. **Scanning**: The scanner processes videos frame-by-frame, detecting faces using the `face_recognition` library
2. **Encoding**: Each face is converted to a 128-dimensional encoding vector
3. **OCR**: On-screen text is extracted during scanning (EasyOCR with Tesseract fallback) and ranked into top fragments
4. **Clustering**: DBSCAN algorithm groups similar face encodings together
5. **Tagging**: Web interface allows manual review and naming of face groups
6. **Manual Review**: Videos without detected faces can be tagged via sampled frame grids
7. **Metadata Planning**: The planner compares DB tags against existing NFO actors and flags risk before writes
8. **NFO Writing**: Jellyfin-compatible NFO files are updated with actor metadata, preserving non-indexium actors
9. **History & Rollback**: Metadata operations are tracked so you can audit writes and roll them back

## Technical Details

- **Face Detection**: HOG or CNN-based face detection from dlib (configurable via `FACE_DETECTION_MODEL`)
- **Face Recognition**: 128-dimensional face encodings for comparison
- **Clustering**: DBSCAN with euclidean distance (eps=0.4, min_samples=5)
- **OCR**: EasyOCR (preferred) with automatic Tesseract fallback
- **Parallel Processing**: Multiprocessing for video scanning
- **File Tracking**: SHA256 hashing (first and last 25 blocks) for file identification
- **Web Framework**: Flask with Tailwind CSS styling
- **NFO Parsing**: lxml for reading/writing Jellyfin NFO XML files

## Troubleshooting

### Common Issues

1. **No videos are discovered**
   - Set `INDEXIUM_VIDEO_DIR` to the correct folder and verify readable video files exist

2. **Videos not being processed**
   - Check file permissions and supported formats (MP4, MKV, MOV, AVI)
   - Ensure videos are not corrupted

3. **Face detection not working well**
   - Adjust `FRAME_SKIP` for more thorough scanning
   - Consider video quality and lighting conditions

4. **Performance issues**
   - Reduce `CPU_CORES` if system becomes unresponsive
   - Increase `FRAME_SKIP` for faster but less thorough scanning

## Safety Features

- **Non-destructive**: Default metadata writes target NFO sidecar files (video files are not rewritten)
- **Graceful shutdown**: Ctrl+C during scanning saves progress before exit
- **File integrity**: Uses content hashing to avoid reprocessing moved files
- **Backup-friendly**: All data stored in single SQLite database file
- **Metadata rollback**: Operation history tracks metadata writes; rollback restores previous NFO state

## Running Tests

```bash
pytest -q                                                    # Run all tests
pytest -q tests/test_scanner.py::test_cluster_faces_updates_ids  # Single test
python e2e_test.py test_vids                                 # End-to-end pipeline test (uses temp DB)
pytest --cov --cov-report=term-missing                       # Tests with coverage
pytest --cov --cov-report=html                               # HTML coverage report in htmlcov/
```
