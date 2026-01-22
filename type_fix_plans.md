# Pyright Type Fix Plan

## Current findings (pyright)
- Total errors: 1190
- Highest-volume files:
  - app.py (210)
  - scanner.py (222)
  - metadata_services.py (35)
  - text_utils.py (11)
  - util.py (1)
  - tests/ (over half the total across multiple files)
- Most common rules:
  - reportUnknownVariableType (411)
  - reportUnknownParameterType (354)
  - reportUnknownMemberType (345)
  - reportArgumentType (49)
- Missing type stubs:
  - easyocr
  - face_recognition
  - ffmpeg
  - pytesseract
  - sklearn.cluster

## Plan of attack

### 1) External library typing strategy (missing stubs)
- Add a `typings/` stub package tree and set `tool.pyright.stubPath` to it.
- Create minimal stubs that cover the surface area actually used by this project for:
  - `easyocr`
  - `face_recognition`
  - `ffmpeg`
  - `pytesseract`
  - `sklearn.cluster`
- Prefer precise annotations for functions/classes the code uses; fall back to `Any` for unsupported portions.
- If a third-party types package exists and is maintained, switch to that instead of local stubs.

### 2) Core app typing (app.py)
- Add return/parameter annotations for Flask handlers and helper functions.
- Type sqlite connections and rows with `sqlite3.Connection`, `sqlite3.Cursor`, and `sqlite3.Row`.
- Replace `dict[str, object]` caches with explicit `TypedDict` structures for `_known_people_cache` and related data.
- Add small helper types (e.g., `SampleInfo`, `KnownPerson`) to reduce `Unknown` leakage.
- Use `typing.cast` only at boundaries where dynamic data is unavoidable (e.g., JSON payloads).

### 3) Scanner pipeline typing (scanner.py)
- Annotate public entry points and pipeline steps (scan, OCR, clustering, file hashing).
- Define types for record tuples returned from sqlite (e.g., `ScannedFileRow`, `FaceRow`).
- Clarify numpy/cv2 types (e.g., `npt.NDArray[np.uint8]`) for frames and embeddings.
- Add typed wrappers around OCR and face-recognition calls to isolate `Any`.

### 4) Metadata services typing (metadata_services.py)
- Add `TypedDict` or `dataclass` models for plan items, history entries, and backup metadata.
- Ensure `MetadataPlanner`, `MetadataWriter`, `BackupManager`, and `HistoryService` are fully annotated.
- Annotate I/O-heavy helpers (file paths, FFmpeg call results) to prevent `Unknown` propagation.

### 5) Text utilities typing (text_utils.py)
- Add explicit types for OCR fragment structures (e.g., `OCRFragment` with score, text, bbox).
- Guard numeric conversions with `int()`/`float()` casts and validation to satisfy operator checks.

### 6) Test typing cleanup (tests/)
- Add explicit fixture types in tests:
  - `tmp_path: Path`
  - `monkeypatch: pytest.MonkeyPatch`
  - `client: flask.testing.FlaskClient`
  - `caplog: pytest.LogCaptureFixture`
- Use `typing.cast` or small helper fixtures where tests intentionally rely on dynamic behavior.
- Keep tests type-friendly without altering test behavior.

### 7) Iterate and tighten
- Re-run `uv run pyright` after each phase to confirm error reductions.
- Address remaining `reportArgumentType` and `reportOperatorIssue` errors last; they usually resolve once core types are in place.
- Keep `typeCheckingMode = "basic"` until errors are <100, then consider raising strictness module-by-module.

## Suggested execution order
1. Add stub path + minimal stubs for missing libraries.
2. Type the database layer and cache structures in `app.py`.
3. Type scanner pipeline entry points and core data types in `scanner.py`.
4. Type metadata services.
5. Type text utilities.
6. Clean up tests.
7. Re-evaluate pyright configuration for stricter checks.
