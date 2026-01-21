# Code Style & Conventions

## General Style
- Python 3.10+ syntax
- No strict type hints enforcement (used sparingly)
- Functions use snake_case
- Classes use PascalCase
- Constants use UPPER_SNAKE_CASE
- Private functions/variables prefixed with underscore (_)

## Configuration Pattern
- All settings centralized in `config.py` via `Config` class
- Settings loaded from environment variables with defaults
- Never hardcode configuration inline - always modify `config.py`

## Database Pattern
- SQLite with connection per-request pattern
- Use `get_db_connection()` / `close_db_connection()` in app.py
- File tracking via SHA256 hash of first+last 25 blocks (content-based, survives rename/move)

## Key Design Patterns
- **Graceful shutdown**: `SignalHandler` class catches Ctrl+C and saves progress
- **OCR fallback**: EasyOCR preferred, falls back to Tesseract if unavailable
- **Caching**: Known people cache in app.py invalidated on tag changes via `_invalidate_known_people_cache()`
- **Multiprocessing**: Scanner uses worker pool based on CPU_CORES setting

## Module Responsibilities
- `app.py` - Web routes, UI, API endpoints
- `scanner.py` - Video processing pipeline, face detection, clustering, OCR
- `metadata_services.py` - Metadata planning/writing with pause/resume/cancel, backups, history
- `config.py` - Centralized configuration
- `text_utils.py` - OCR text fragment ranking/filtering
- `util.py` - File hashing utilities
- `signal_handler.py` - Graceful shutdown handling

## Test Conventions
- Tests in `tests/` directory
- Use monkeypatching to redirect DB/paths to temp locations
- Shared fixtures in `tests/conftest.py`
- E2E tests use Playwright for browser automation
