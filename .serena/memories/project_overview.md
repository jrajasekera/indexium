# Indexium - Project Overview

## Purpose
Indexium is a Python-based video face scanning and tagging application. It:
- Detects faces in videos using ML
- Clusters faces by person using DBSCAN
- Provides a web UI for tagging faces and managing metadata
- Writes "People: Name1, Name2" metadata to video files via ffmpeg

## Tech Stack
- **Language**: Python 3.10+
- **Package Manager**: `uv` (preferred) or pip
- **Web Framework**: Flask 3.0.3
- **Database**: SQLite (file: `video_faces.db`)
- **ML/CV Libraries**:
  - `face-recognition` 1.3.0 - Face detection and encoding (128-dim vectors)
  - `opencv-python` 4.9.0.80 - Video processing
  - `scikit-learn` 1.5.0 - DBSCAN clustering
- **OCR**: EasyOCR (primary), Tesseract (fallback)
- **Testing**: pytest, Playwright (E2E browser tests)
- **Other**: numpy, Pillow, ffmpeg-python, python-dotenv

## Key Entry Points
- `app.py` - Flask web UI (runs on http://localhost:5001)
- `scanner.py` - CLI for video scanning and OCR operations

## Database Schema (SQLite)
- `scanned_files` - Processed videos by content hash, face counts, manual review status
- `faces` - Face data with locations, 128-dim encodings, cluster IDs, person tags
- `video_people` - Links videos-without-faces to manually assigned person tags
- `video_text` - OCR text snippets keyed by video hash
