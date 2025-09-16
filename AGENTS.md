# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Flask web UI for tagging, reviewing, and metadata writing.
- `scanner.py`: Video scanning, face detection/encoding, clustering, auto-classify.
- `config.py`: Env-driven settings (paths, thresholds, cores). Centralize changes here.
- `util.py`, `signal_handler.py`: Helpers and graceful shutdown.
- `tests/`: Pytest suite; see `test_app.py`, `test_scanner.py`, etc.
- `templates/`: Jinja2 templates for the UI.
- `thumbnails/`, `video_faces.db`: Generated at runtime (gitignored).

## Build, Test, and Development Commands
- Install deps (preferred): `uv sync`  • Alt: `pip install -e .`
- Run scanner: `INDEXIUM_VIDEO_DIR=/path python scanner.py`
- Start web UI: `python app.py` → http://localhost:5001
- Run tests: `pytest -q`  • Example single test: `pytest -q tests/test_scanner.py::test_cluster_faces_updates_ids`
- End-to-end check: `python e2e_test.py test_vids` (creates temp DB/thumbs).

## Coding Style & Naming Conventions
- Python 3.10+. Follow PEP 8 (4-space indents, 100–120 col soft limit).
- Use snake_case for functions/variables, PascalCase for classes, module names in lowercase.
- Add concise docstrings for public functions and routes; prefer type hints where practical.
- Keep functions small and side-effect-aware; prefer explicit over implicit configuration via `config.py`.

## Testing Guidelines
- Framework: Pytest. Place tests under `tests/` as `test_*.py`; name tests `test_*`.
- Use fixtures/monkeypatching to point DB/paths to temp locations (see existing tests).
- Run locally with `pytest -q`; target specific tests during development for speed.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, concise summary (e.g., "Add pagination for tag_group"). Optional scope tags (e.g., `test:`) are welcome.
- PRs should include: clear description, rationale, testing steps, and screenshots/GIFs for UI changes.
- Link related issues; keep diffs focused. Update `README.md`/`config.py` docstrings when adding new settings.

## Security & Configuration Tips
- Do not commit generated assets: `video_faces.db`, `thumbnails/` (already in `.gitignore`).
- Required tools: ffmpeg, OpenCV/dlib system deps (see README). Ensure env vars like `INDEXIUM_VIDEO_DIR`, `INDEXIUM_DB`, `FLASK_DEBUG` are set as needed.

