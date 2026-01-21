# Codebase Architecture

## Directory Structure
```
indexium/
├── app.py                 # Flask web application (main UI entry point)
├── scanner.py             # Video processing CLI (face detection, clustering, OCR)
├── metadata_services.py   # Metadata planning/writing system
├── config.py              # Centralized configuration
├── text_utils.py          # OCR text fragment utilities
├── util.py                # File hashing utility
├── signal_handler.py      # Graceful shutdown handler
├── e2e_test.py            # End-to-end pipeline test
├── templates/             # Jinja2 HTML templates
│   ├── base.html          # Base layout
│   ├── group_tagger.html  # Face cluster tagging UI
│   ├── people_list.html   # People listing
│   ├── person_detail.html # Individual person view
│   ├── metadata_*.html    # Metadata management views
│   └── video_manual_*.html # Manual video review views
├── tests/                 # pytest test suite
│   ├── conftest.py        # Shared fixtures
│   ├── test_app.py        # Flask routes tests
│   ├── test_scanner.py    # Scanner logic tests
│   ├── test_metadata_*.py # Metadata service tests
│   ├── test_e2e*.py       # End-to-end tests
│   └── test_*.py          # Other unit tests
├── thumbnails/            # Generated face thumbnails
├── test_vids/             # Test video files
└── video_faces.db         # SQLite database
```

## Processing Pipeline Flow
1. **Scanner** processes videos frame-by-frame
2. **Face encodings** extracted (128-dimensional vectors)
3. **DBSCAN clustering** groups similar faces (eps=0.4, min_samples=5)
4. **Web UI** presents clusters for manual naming
5. **Metadata writer** embeds "People: Name1, Name2" into video file comments via ffmpeg

## Key Classes

### MetadataServices (metadata_services.py)
- `MetadataPlanner` - Generates plans comparing DB tags vs file metadata
- `MetadataWriter` - Executes writes with pause/resume/cancel support
- `BackupManager` - Creates/restores file backups before writes
- `HistoryService` - Tracks operation history for rollback

### SignalHandler (signal_handler.py)
- Catches SIGINT/SIGTERM for graceful shutdown
- Saves progress before exit

### Config (config.py)
- Centralized settings via environment variables
- All configuration loaded at import time
