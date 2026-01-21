# Suggested Commands

## Setup & Installation
```bash
uv sync                          # Install dependencies (preferred)
pip install -e .                 # Alternative: editable install
```

## Running the Application
```bash
python app.py                    # Start web UI at http://localhost:5001
INDEXIUM_VIDEO_DIR=/path python scanner.py  # Run face scanner
```

## Scanner CLI Commands
```bash
python scanner.py                        # Full video scan
python scanner.py refresh_ocr            # Refresh OCR for all completed videos
python scanner.py refresh_ocr HASH123    # Refresh specific file by hash
python scanner.py continue_ocr           # Process videos missing OCR
python scanner.py cleanup_ocr            # Remove short OCR text (<4 chars)
python scanner.py cleanup_ocr 6          # Custom minimum length
```

## Testing
```bash
pytest -q                        # Run all tests
pytest -q tests/test_scanner.py  # Single test file
pytest -q tests/test_scanner.py::test_cluster_faces_updates_ids  # Single test
python e2e_test.py test_vids     # End-to-end pipeline test (uses temp DB)
```

## Key Environment Variables
- `INDEXIUM_VIDEO_DIR` - Video source directory
- `INDEXIUM_DB` - SQLite database path (default: `video_faces.db`)
- `FRAME_SKIP` - Frames between face detection samples (default: 25)
- `CPU_CORES` - Worker processes (default: all cores)
- `INDEXIUM_OCR_ENABLED` - Toggle OCR (default: true)
- `INDEXIUM_OCR_ENGINE` - `easyocr`, `tesseract`, or `auto`

## System Utils (macOS/Darwin)
```bash
git status / git diff / git log   # Git operations
ls -la                            # List files
find . -name "*.py"               # Find files
grep -r "pattern" .               # Search in files
```
