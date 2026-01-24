# NFO Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace video file comment-based metadata with Jellyfin-compatible NFO files using `<actor source="indexium">` tags.

**Architecture:** New `nfo_services.py` module with NfoService (XML read/write), NfoBackupManager, NfoPlanner, NfoWriter, and NfoHistoryService. Maintains 1:1 video-to-plan-item mapping with shared NFO write coordination at execution time.

**Tech Stack:** Python 3.10+, lxml for XML parsing, SQLite, Flask, existing test infrastructure (pytest)

---

## Phase 1: Test Fixtures and Core Infrastructure

### Task 1: Create NFO Test Fixtures

**Files:**
- Create: `tests/fixtures/nfo/empty_actors.nfo`
- Create: `tests/fixtures/nfo/existing_actors.nfo`
- Create: `tests/fixtures/nfo/indexium_actors.nfo`
- Create: `tests/fixtures/nfo/mixed_actors.nfo`
- Create: `tests/fixtures/nfo/malformed.nfo`

**Step 1: Create fixtures directory**

```bash
mkdir -p tests/fixtures/nfo
```

**Step 2: Create empty_actors.nfo**

```xml
<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<movie>
  <title>Empty Test Video</title>
  <plot />
</movie>
```

**Step 3: Create existing_actors.nfo (TMDb-style actors)**

```xml
<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<movie>
  <title>Test Video With Actors</title>
  <actor>
    <name>Tom Hanks</name>
    <role>Forrest</role>
    <type>Actor</type>
    <thumb>https://image.tmdb.org/t/p/original/abc.jpg</thumb>
  </actor>
  <actor>
    <name>Robin Wright</name>
    <role>Jenny</role>
    <type>Actor</type>
  </actor>
</movie>
```

**Step 4: Create indexium_actors.nfo**

```xml
<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<movie>
  <title>Test Video With Indexium Actors</title>
  <actor source="indexium">
    <name>John Smith</name>
  </actor>
  <actor source="indexium">
    <name>Jane Doe</name>
  </actor>
</movie>
```

**Step 5: Create mixed_actors.nfo**

```xml
<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<movie>
  <title>Test Video Mixed Actors</title>
  <actor>
    <name>Tom Hanks</name>
    <role>Forrest</role>
    <type>Actor</type>
  </actor>
  <actor source="indexium">
    <name>John Smith</name>
  </actor>
</movie>
```

**Step 6: Create malformed.nfo**

```xml
<?xml version="1.0" encoding="utf-8"?>
<movie>
  <title>Malformed XML</title>
  <actor>
    <name>Unclosed Tag
  </actor>
</movie>
```

**Step 7: Commit fixtures**

```bash
git add tests/fixtures/nfo/
git commit -m "feat: add NFO test fixtures for metadata tests"
```

---

### Task 2: Add lxml Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add lxml to dependencies**

In `pyproject.toml`, add `lxml` to the dependencies list:

```toml
dependencies = [
    # ... existing deps ...
    "lxml>=5.0.0",
]
```

**Step 2: Install dependencies**

```bash
uv sync
```

**Step 3: Verify lxml is available**

```bash
python -c "from lxml import etree; print('lxml OK')"
```
Expected: `lxml OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add lxml dependency for NFO XML parsing"
```

---

### Task 3: Add NFO Config Options

**Files:**
- Modify: `config.py`
- Test: `tests/test_config.py`

**Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
def test_nfo_config_defaults():
    """Test NFO-related config defaults."""
    from config import Config
    cfg = Config()
    assert cfg.NFO_REMOVE_STALE_ACTORS is True
    assert cfg.NFO_BACKUP_MAX_AGE_DAYS == 30
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py::test_nfo_config_defaults -v
```
Expected: FAIL with AttributeError

**Step 3: Add config options to config.py**

Find the `Config` dataclass and add:

```python
    # NFO Metadata Settings
    NFO_REMOVE_STALE_ACTORS: bool = field(
        default_factory=lambda: env_bool("NFO_REMOVE_STALE_ACTORS", True)
    )
    NFO_BACKUP_MAX_AGE_DAYS: int = field(
        default_factory=lambda: env_int("NFO_BACKUP_MAX_AGE_DAYS", 30)
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py::test_nfo_config_defaults -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add NFO_REMOVE_STALE_ACTORS and NFO_BACKUP_MAX_AGE_DAYS config"
```

---

## Phase 2: Core NfoService

### Task 4: Create NfoParseError and NfoActor Classes

**Files:**
- Create: `nfo_services.py`
- Create: `tests/test_nfo_services.py`

**Step 1: Write failing test for NfoParseError**

Create `tests/test_nfo_services.py`:

```python
"""Tests for NFO metadata services."""

import pytest


def test_nfo_parse_error_has_path_and_reason():
    """NfoParseError stores path and reason."""
    from nfo_services import NfoParseError

    err = NfoParseError("/path/to/file.nfo", "XML syntax error")
    assert err.path == "/path/to/file.nfo"
    assert err.reason == "XML syntax error"
    assert "file.nfo" in str(err)
    assert "XML syntax error" in str(err)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_nfo_services.py::test_nfo_parse_error_has_path_and_reason -v
```
Expected: FAIL (module not found)

**Step 3: Create nfo_services.py with NfoParseError**

Create `nfo_services.py`:

```python
"""NFO metadata services for Jellyfin-compatible metadata management."""

from __future__ import annotations


class NfoParseError(Exception):
    """Raised when NFO file cannot be parsed."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse {path}: {reason}")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_nfo_services.py::test_nfo_parse_error_has_path_and_reason -v
```
Expected: PASS

**Step 5: Write test for NfoActor**

Add to `tests/test_nfo_services.py`:

```python
def test_nfo_actor_dataclass():
    """NfoActor stores actor data."""
    from nfo_services import NfoActor

    actor = NfoActor(name="John Smith", source="indexium", role="Self")
    assert actor.name == "John Smith"
    assert actor.source == "indexium"
    assert actor.role == "Self"
    assert actor.type is None
    assert actor.thumb is None
    assert actor.raw_element is None
```

**Step 6: Run test to verify it fails**

```bash
pytest tests/test_nfo_services.py::test_nfo_actor_dataclass -v
```
Expected: FAIL

**Step 7: Add NfoActor dataclass**

Add to `nfo_services.py`:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NfoActor:
    """Actor data extracted from NFO file.

    Note: raw_element is NOT cached. When writing, we always read the NFO
    fresh to get current raw elements, ensuring unknown children are preserved.
    """

    name: str
    source: str | None = None
    role: str | None = None
    type: str | None = None
    thumb: str | None = None
    raw_element: Any = field(default=None, repr=False)  # lxml Element, not cached
```

**Step 8: Run test to verify it passes**

```bash
pytest tests/test_nfo_services.py::test_nfo_actor_dataclass -v
```
Expected: PASS

**Step 9: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: add NfoParseError and NfoActor classes"
```

---

### Task 5: Implement NfoService.find_nfo_path

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing tests**

Add to `tests/test_nfo_services.py`:

```python
def test_find_nfo_path_exact_match(tmp_path):
    """find_nfo_path returns video-specific NFO when it exists."""
    from nfo_services import NfoService

    video = tmp_path / "video.mp4"
    nfo = tmp_path / "video.nfo"
    video.touch()
    nfo.write_text("<movie></movie>")

    service = NfoService()
    result = service.find_nfo_path(str(video))
    assert result == str(nfo)


def test_find_nfo_path_case_insensitive(tmp_path):
    """find_nfo_path finds .NFO (uppercase) variant."""
    from nfo_services import NfoService

    video = tmp_path / "video.mp4"
    nfo = tmp_path / "video.NFO"
    video.touch()
    nfo.write_text("<movie></movie>")

    service = NfoService()
    result = service.find_nfo_path(str(video))
    assert result == str(nfo)


def test_find_nfo_path_movie_nfo_fallback(tmp_path):
    """find_nfo_path falls back to movie.nfo."""
    from nfo_services import NfoService

    video = tmp_path / "video.mp4"
    nfo = tmp_path / "movie.nfo"
    video.touch()
    nfo.write_text("<movie></movie>")

    service = NfoService()
    result = service.find_nfo_path(str(video))
    assert result == str(nfo)


def test_find_nfo_path_missing_returns_none(tmp_path):
    """find_nfo_path returns None when no NFO exists."""
    from nfo_services import NfoService

    video = tmp_path / "video.mp4"
    video.touch()

    service = NfoService()
    result = service.find_nfo_path(str(video))
    assert result is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py::test_find_nfo_path_exact_match -v
pytest tests/test_nfo_services.py::test_find_nfo_path_case_insensitive -v
pytest tests/test_nfo_services.py::test_find_nfo_path_movie_nfo_fallback -v
pytest tests/test_nfo_services.py::test_find_nfo_path_missing_returns_none -v
```
Expected: All FAIL

**Step 3: Implement NfoService class with find_nfo_path**

Add to `nfo_services.py`:

```python
from pathlib import Path


class NfoService:
    """Core service for reading and writing NFO files."""

    def find_nfo_path(self, video_path: str) -> str | None:
        """Find NFO file for a video, checking multiple naming conventions.

        Checks in order (returns first match):
        1. <video_name>.nfo / .NFO (video-specific, preferred)
        2. movie.nfo / Movie.nfo (shared, may apply to multiple videos)
        """
        base = Path(video_path).stem
        parent = Path(video_path).parent

        candidates = [
            parent / f"{base}.nfo",
            parent / f"{base}.NFO",
            parent / "movie.nfo",
            parent / "Movie.nfo",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "find_nfo_path" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoService.find_nfo_path"
```

---

### Task 6: Implement NfoService.read_actors

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing tests**

Add to `tests/test_nfo_services.py`:

```python
import shutil
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "nfo"


def test_read_actors_parses_all_actors(tmp_path):
    """read_actors returns all actors from NFO."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "existing_actors.nfo", nfo)

    service = NfoService()
    actors = service.read_actors(str(nfo))

    assert len(actors) == 2
    names = {a.name for a in actors}
    assert names == {"Tom Hanks", "Robin Wright"}


def test_read_actors_captures_source_attribute(tmp_path):
    """read_actors captures source='indexium' attribute."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "indexium_actors.nfo", nfo)

    service = NfoService()
    actors = service.read_actors(str(nfo))

    assert len(actors) == 2
    assert all(a.source == "indexium" for a in actors)


def test_read_actors_preserves_full_structure(tmp_path):
    """read_actors preserves role, type, thumb fields."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "existing_actors.nfo", nfo)

    service = NfoService()
    actors = service.read_actors(str(nfo))

    tom = next(a for a in actors if a.name == "Tom Hanks")
    assert tom.role == "Forrest"
    assert tom.type == "Actor"
    assert tom.thumb is not None
    assert tom.raw_element is not None


def test_read_actors_malformed_xml_raises(tmp_path):
    """read_actors raises NfoParseError for malformed XML."""
    from nfo_services import NfoService, NfoParseError

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "malformed.nfo", nfo)

    service = NfoService()
    with pytest.raises(NfoParseError) as exc_info:
        service.read_actors(str(nfo))

    assert str(nfo) in str(exc_info.value)


def test_read_actors_empty_returns_empty_list(tmp_path):
    """read_actors returns empty list when no actors."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "empty_actors.nfo", nfo)

    service = NfoService()
    actors = service.read_actors(str(nfo))

    assert actors == []
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py -k "read_actors" -v
```
Expected: All FAIL

**Step 3: Implement read_actors**

Add to `NfoService` class in `nfo_services.py`:

```python
from lxml import etree


class NfoService:
    # ... existing find_nfo_path ...

    def read_actors(self, nfo_path: str) -> list[NfoActor]:
        """Parse NFO XML, return all actors with full structure preserved.

        Raises:
            NfoParseError: If XML is malformed or unreadable
        """
        root, _ = self._read_xml(nfo_path)
        actors = []

        for actor_elem in root.findall("actor"):
            name_elem = actor_elem.find("name")
            if name_elem is None or not name_elem.text:
                continue

            actors.append(
                NfoActor(
                    name=name_elem.text.strip(),
                    source=actor_elem.get("source"),
                    role=self._get_child_text(actor_elem, "role"),
                    type=self._get_child_text(actor_elem, "type"),
                    thumb=self._get_child_text(actor_elem, "thumb"),
                    raw_element=actor_elem,
                )
            )

        return actors

    def _read_xml(self, nfo_path: str) -> tuple[etree._Element, str | None]:
        """Read NFO file, return (root element, detected encoding)."""
        try:
            with open(nfo_path, "rb") as f:
                raw = f.read()
        except OSError as e:
            raise NfoParseError(nfo_path, f"Cannot read file: {e}")

        # Detect BOM and encoding
        encoding = None
        if raw.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
            raw = raw[3:]
            encoding = "utf-8-sig"

        try:
            parser = etree.XMLParser(remove_blank_text=False, recover=True)
            root = etree.fromstring(raw, parser)

            if root is None:
                raise NfoParseError(nfo_path, "XML recovery failed")

            return root, encoding

        except etree.XMLSyntaxError as e:
            raise NfoParseError(nfo_path, f"XML syntax error: {e}")

    @staticmethod
    def _get_child_text(elem: etree._Element, tag: str) -> str | None:
        """Get text content of child element, or None."""
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "read_actors" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoService.read_actors with XML parsing"
```

---

### Task 7: Implement NfoService.write_actors

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing tests**

Add to `tests/test_nfo_services.py`:

```python
def test_write_actors_adds_indexium_actors(tmp_path):
    """write_actors adds new actors with source='indexium'."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "empty_actors.nfo", nfo)

    service = NfoService()
    service.write_actors(str(nfo), ["Alice", "Bob"])

    # Re-read and verify
    actors = service.read_actors(str(nfo))
    assert len(actors) == 2
    assert {a.name for a in actors} == {"Alice", "Bob"}
    assert all(a.source == "indexium" for a in actors)


def test_write_actors_preserves_non_indexium_actors(tmp_path):
    """write_actors preserves existing non-indexium actors."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "existing_actors.nfo", nfo)

    service = NfoService()
    service.write_actors(str(nfo), ["Alice"])

    actors = service.read_actors(str(nfo))
    names = {a.name for a in actors}
    # Original actors preserved
    assert "Tom Hanks" in names
    assert "Robin Wright" in names
    # New actor added
    assert "Alice" in names
    assert len(actors) == 3


def test_write_actors_replaces_indexium_actors(tmp_path):
    """write_actors replaces existing indexium actors."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "indexium_actors.nfo", nfo)

    service = NfoService()
    service.write_actors(str(nfo), ["Alice", "Bob"])

    actors = service.read_actors(str(nfo))
    indexium_actors = [a for a in actors if a.source == "indexium"]
    assert len(indexium_actors) == 2
    assert {a.name for a in indexium_actors} == {"Alice", "Bob"}
    # Old indexium actors removed
    assert "John Smith" not in {a.name for a in actors}


def test_write_actors_preserves_mixed_actors(tmp_path):
    """write_actors in mixed scenario: replaces indexium, preserves others."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "mixed_actors.nfo", nfo)

    service = NfoService()
    service.write_actors(str(nfo), ["Alice"])

    actors = service.read_actors(str(nfo))
    # Tom Hanks (non-indexium) preserved
    assert any(a.name == "Tom Hanks" and a.source is None for a in actors)
    # John Smith (old indexium) removed, Alice (new indexium) added
    assert any(a.name == "Alice" and a.source == "indexium" for a in actors)
    assert not any(a.name == "John Smith" for a in actors)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py -k "write_actors" -v
```
Expected: All FAIL

**Step 3: Implement write_actors**

Add to `NfoService` class in `nfo_services.py`:

```python
    def write_actors(
        self,
        nfo_path: str,
        indexium_actors: list[str],
        preserve_existing: bool = True,
    ) -> None:
        """Update NFO: keep non-indexium actors, replace indexium actors.

        Raises:
            NfoParseError: If XML is malformed
        """
        root, encoding = self._read_xml(nfo_path)

        # Remove existing indexium actors
        for actor_elem in root.findall("actor"):
            if actor_elem.get("source") == "indexium":
                root.remove(actor_elem)

        # Add new indexium actors
        for name in sorted(set(indexium_actors), key=str.lower):
            actor_elem = etree.SubElement(root, "actor", source="indexium")
            name_elem = etree.SubElement(actor_elem, "name")
            name_elem.text = name

        self._write_xml(nfo_path, root, encoding)

    def _write_xml(
        self, nfo_path: str, root: etree._Element, encoding: str | None
    ) -> None:
        """Write NFO file, preserving original encoding."""
        xml_bytes = etree.tostring(
            root,
            encoding=encoding or "utf-8",
            xml_declaration=True,
            pretty_print=False,
        )

        with open(nfo_path, "wb") as f:
            if encoding == "utf-8-sig":
                f.write(b"\xef\xbb\xbf")  # Write BOM
            f.write(xml_bytes)

    def get_indexium_actors(self, nfo_path: str) -> list[str]:
        """Convenience: return only names where source='indexium'."""
        actors = self.read_actors(nfo_path)
        return [a.name for a in actors if a.source == "indexium"]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "write_actors" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoService.write_actors"
```

---

## Phase 3: Backup Manager

### Task 8: Implement NfoBackupManager

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing tests**

Add to `tests/test_nfo_services.py`:

```python
def test_backup_manager_create_backup(tmp_path):
    """create_backup copies NFO to operation-scoped backup."""
    from nfo_services import NfoBackupManager

    nfo = tmp_path / "video.nfo"
    nfo.write_text("<movie><title>Test</title></movie>")

    manager = NfoBackupManager()
    backup_path = manager.create_backup(str(nfo), operation_id=42)

    assert backup_path == str(nfo) + ".bak.42"
    assert Path(backup_path).exists()
    assert Path(backup_path).read_text() == nfo.read_text()


def test_backup_manager_restore_backup(tmp_path):
    """restore_backup restores from operation-scoped backup."""
    from nfo_services import NfoBackupManager

    nfo = tmp_path / "video.nfo"
    original_content = "<movie><title>Original</title></movie>"
    nfo.write_text(original_content)

    manager = NfoBackupManager()
    manager.create_backup(str(nfo), operation_id=42)

    # Modify the NFO
    nfo.write_text("<movie><title>Modified</title></movie>")

    # Restore
    result = manager.restore_backup(str(nfo), operation_id=42)
    assert result is True
    assert nfo.read_text() == original_content


def test_backup_manager_restore_missing_returns_false(tmp_path):
    """restore_backup returns False when backup doesn't exist."""
    from nfo_services import NfoBackupManager

    nfo = tmp_path / "video.nfo"
    nfo.write_text("<movie></movie>")

    manager = NfoBackupManager()
    result = manager.restore_backup(str(nfo), operation_id=999)
    assert result is False


def test_backup_manager_cleanup_backup(tmp_path):
    """cleanup_backup removes the backup file."""
    from nfo_services import NfoBackupManager

    nfo = tmp_path / "video.nfo"
    nfo.write_text("<movie></movie>")

    manager = NfoBackupManager()
    backup_path = manager.create_backup(str(nfo), operation_id=42)
    assert Path(backup_path).exists()

    manager.cleanup_backup(str(nfo), operation_id=42)
    assert not Path(backup_path).exists()


def test_backup_manager_find_backup_path():
    """find_backup_path returns expected path format."""
    from nfo_services import NfoBackupManager

    manager = NfoBackupManager()
    path = manager.find_backup_path("/path/to/video.nfo", operation_id=123)
    assert path == "/path/to/video.nfo.bak.123"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py -k "backup_manager" -v
```
Expected: All FAIL

**Step 3: Implement NfoBackupManager**

Add to `nfo_services.py`:

```python
import shutil


class NfoBackupManager:
    """Operation-scoped backup/restore for NFO files."""

    def find_backup_path(self, nfo_path: str, operation_id: int) -> str:
        """Return expected backup path for given NFO and operation."""
        return f"{nfo_path}.bak.{operation_id}"

    def create_backup(self, nfo_path: str, operation_id: int) -> str:
        """Copy NFO to operation-scoped backup, return backup path."""
        backup_path = self.find_backup_path(nfo_path, operation_id)
        shutil.copy2(nfo_path, backup_path)
        return backup_path

    def restore_backup(self, nfo_path: str, operation_id: int) -> bool:
        """Restore from operation-specific backup if exists. Returns success."""
        backup_path = self.find_backup_path(nfo_path, operation_id)
        if not Path(backup_path).exists():
            return False
        shutil.copy2(backup_path, nfo_path)
        return True

    def cleanup_backup(self, nfo_path: str, operation_id: int) -> None:
        """Remove operation-specific backup file."""
        backup_path = self.find_backup_path(nfo_path, operation_id)
        try:
            Path(backup_path).unlink()
        except FileNotFoundError:
            pass

    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Remove backup files older than max_age_days. Returns count removed."""
        # This would scan for .bak.* files and check mtime
        # Implementation deferred - needs directory scanning
        return 0
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "backup_manager" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoBackupManager with operation-scoped backups"
```

---

## Phase 4: Schema Migration

### Task 9: Add NFO Schema to scanner.py

**Files:**
- Modify: `scanner.py`
- Modify: `tests/test_scanner.py`

**Step 1: Write failing test**

Add to `tests/test_scanner.py`:

```python
def test_nfo_actor_cache_table_created(tmp_path, monkeypatch):
    """ensure_db_schema creates nfo_actor_cache table."""
    import sqlite3
    from scanner import ensure_db_schema

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))

    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)

    # Check table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nfo_actor_cache'"
    )
    assert cursor.fetchone() is not None

    # Check columns
    cursor = conn.execute("PRAGMA table_info(nfo_actor_cache)")
    columns = {row[1] for row in cursor.fetchall()}
    assert "nfo_path" in columns
    assert "actors_json" in columns
    assert "nfo_mtime" in columns

    conn.close()


def test_nfo_path_column_added_to_operation_items(tmp_path, monkeypatch):
    """ensure_db_schema adds nfo_path column to metadata_operation_items."""
    import sqlite3
    from scanner import ensure_db_schema

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))

    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)

    cursor = conn.execute("PRAGMA table_info(metadata_operation_items)")
    columns = {row[1] for row in cursor.fetchall()}
    assert "nfo_path" in columns

    conn.close()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scanner.py::test_nfo_actor_cache_table_created -v
pytest tests/test_scanner.py::test_nfo_path_column_added_to_operation_items -v
```
Expected: FAIL

**Step 3: Add schema changes to scanner.py**

Find `ensure_db_schema()` in `scanner.py` and add after existing table creation:

```python
        # NFO actor cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nfo_actor_cache (
                nfo_path TEXT PRIMARY KEY,
                actors_json TEXT,
                nfo_mtime REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add nfo_path column to metadata_operation_items if missing
        cursor.execute("PRAGMA table_info(metadata_operation_items)")
        columns = {row[1] for row in cursor.fetchall()}
        if "nfo_path" not in columns:
            cursor.execute(
                "ALTER TABLE metadata_operation_items ADD COLUMN nfo_path TEXT"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_items_nfo_path "
                "ON metadata_operation_items (nfo_path)"
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scanner.py::test_nfo_actor_cache_table_created -v
pytest tests/test_scanner.py::test_nfo_path_column_added_to_operation_items -v
```
Expected: PASS

**Step 5: Run full test suite to ensure no regressions**

```bash
pytest -q
```
Expected: All tests pass

**Step 6: Commit**

```bash
git add scanner.py tests/test_scanner.py
git commit -m "feat: add nfo_actor_cache table and nfo_path column migration"
```

---

## Phase 5: NfoPlanItem Data Class

### Task 10: Implement NfoPlanItem

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing test**

Add to `tests/test_nfo_services.py`:

```python
def test_nfo_plan_item_has_all_ui_fields():
    """NfoPlanItem has all fields needed by UI."""
    from nfo_services import NfoPlanItem

    item = NfoPlanItem(
        file_hash="abc123",
        file_path="/path/to/video.mp4",
        file_name="video.mp4",
        file_extension=".mp4",
        nfo_path="/path/to/video.nfo",
        db_people=["Alice", "Bob"],
        existing_people=["Charlie"],
        result_people=["Alice", "Bob"],
        tags_to_add=["Alice", "Bob"],
        tags_to_remove=["Charlie"],
        existing_indexium_actors=["Charlie"],
        other_actors=[],
        existing_comment="People: Charlie",
        result_comment="People: Alice, Bob",
        risk_level="warning",
        can_update=True,
    )

    # Check all UI-required fields exist
    assert item.file_hash == "abc123"
    assert item.file_path == "/path/to/video.mp4"
    assert item.file_name == "video.mp4"
    assert item.file_extension == ".mp4"
    assert item.nfo_path == "/path/to/video.nfo"
    assert item.tag_count == 0  # default
    assert item.new_tag_count == 0  # default
    assert item.issues == []
    assert item.issue_codes == []
    assert item.metadata_only_people == []


def test_nfo_plan_item_requires_update():
    """NfoPlanItem.requires_update reflects actual need."""
    from nfo_services import NfoPlanItem

    # Needs update: has tags to add
    item1 = NfoPlanItem(
        file_hash="a", file_path=None, file_name=None, file_extension=None,
        nfo_path="/test.nfo", db_people=["Alice"], existing_people=[],
        result_people=["Alice"], tags_to_add=["Alice"], tags_to_remove=[],
        existing_indexium_actors=[], other_actors=[],
        existing_comment="", result_comment="People: Alice",
        risk_level="safe", can_update=True,
    )
    assert item1.requires_update is True

    # No update: can_update is False
    item2 = NfoPlanItem(
        file_hash="b", file_path=None, file_name=None, file_extension=None,
        nfo_path=None, db_people=["Alice"], existing_people=[],
        result_people=["Alice"], tags_to_add=["Alice"], tags_to_remove=[],
        existing_indexium_actors=[], other_actors=[],
        existing_comment="", result_comment="People: Alice",
        risk_level="blocked", can_update=False,
    )
    assert item2.requires_update is False

    # No update: nothing to change
    item3 = NfoPlanItem(
        file_hash="c", file_path=None, file_name=None, file_extension=None,
        nfo_path="/test.nfo", db_people=[], existing_people=[],
        result_people=[], tags_to_add=[], tags_to_remove=[],
        existing_indexium_actors=[], other_actors=[],
        existing_comment="", result_comment="",
        risk_level="safe", can_update=True,
    )
    assert item3.requires_update is False


def test_nfo_plan_item_to_dict():
    """NfoPlanItem.to_dict serializes for API."""
    from nfo_services import NfoPlanItem, NfoActor

    item = NfoPlanItem(
        file_hash="abc123",
        file_path="/path/to/video.mp4",
        file_name="video.mp4",
        file_extension=".mp4",
        nfo_path="/path/to/video.nfo",
        db_people=["Alice"],
        existing_people=[],
        result_people=["Alice"],
        tags_to_add=["Alice"],
        tags_to_remove=[],
        existing_indexium_actors=[],
        other_actors=[NfoActor(name="Tom", source=None)],
        existing_comment="",
        result_comment="People: Alice",
        risk_level="safe",
        can_update=True,
    )

    d = item.to_dict()
    assert d["file_hash"] == "abc123"
    assert d["requires_update"] is True
    # Internal fields should be removed
    assert "other_actors" not in d
    assert "existing_indexium_actors" not in d
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py -k "nfo_plan_item" -v
```
Expected: All FAIL

**Step 3: Implement NfoPlanItem**

Add to `nfo_services.py`:

```python
from dataclasses import asdict


@dataclass
class NfoPlanItem:
    """Plan item with full UI/API compatibility."""

    # Core identifiers
    file_hash: str
    file_path: str | None
    file_name: str | None
    file_extension: str | None

    # NFO-specific
    nfo_path: str | None

    # People data
    db_people: list[str]
    existing_people: list[str]
    result_people: list[str]
    tags_to_add: list[str]
    tags_to_remove: list[str]

    # NFO-specific internal data
    existing_indexium_actors: list[str]
    other_actors: list[NfoActor]

    # Comment fields (for UI compatibility)
    existing_comment: str | None
    result_comment: str

    # Status and risk
    risk_level: str
    can_update: bool

    # Issues tracking
    issues: list[str] = field(default_factory=list)
    issue_codes: list[str] = field(default_factory=list)
    probe_error: str | None = None

    # UI display fields
    metadata_only_people: list[str] = field(default_factory=list)
    will_overwrite_comment: bool = False
    overwrites_custom_comment: bool = False
    tag_count: int = 0
    new_tag_count: int = 0
    file_modified_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON API response."""
        data = asdict(self)
        data["requires_update"] = self.requires_update
        # Remove internal fields not needed by UI
        data.pop("other_actors", None)
        data.pop("existing_indexium_actors", None)
        return data

    @property
    def requires_update(self) -> bool:
        """True if this item needs to be written."""
        if not self.can_update:
            return False
        return bool(self.tags_to_add or self.tags_to_remove)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "nfo_plan_item" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoPlanItem with UI/API compatibility"
```

---

## Phase 6: NfoPlanner (Core Planning Logic)

### Task 11: Implement NfoPlanner.build_plan (Basic)

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing tests**

Add to `tests/test_nfo_services.py`:

```python
def test_planner_builds_plan_for_video_with_nfo(tmp_path, monkeypatch):
    """NfoPlanner builds plan item for video with NFO."""
    import sqlite3
    from nfo_services import NfoPlanner
    from scanner import ensure_db_schema

    # Set up DB
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))
    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)

    # Create video and NFO
    video = tmp_path / "video.mp4"
    video.touch()
    nfo = tmp_path / "video.nfo"
    shutil.copy(FIXTURES_DIR / "empty_actors.nfo", nfo)

    # Add to scanned_files
    file_hash = "test_hash_123"
    conn.execute(
        "INSERT INTO scanned_files (file_hash, file_path, file_name, face_count, status) "
        "VALUES (?, ?, ?, 1, 'completed')",
        (file_hash, str(video), "video.mp4"),
    )
    # Add a face with person tag
    conn.execute(
        "INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name) "
        "VALUES (?, 0, '0,0,100,100', ?, 'Alice')",
        (file_hash, "0" * 256),  # Dummy encoding
    )
    conn.commit()

    # Build plan
    planner = NfoPlanner(str(db_path))
    items = planner.build_plan([file_hash])

    assert len(items) == 1
    item = items[0]
    assert item.file_hash == file_hash
    assert item.nfo_path == str(nfo)
    assert item.can_update is True
    assert "Alice" in item.db_people

    conn.close()


def test_planner_marks_missing_nfo_as_blocked(tmp_path, monkeypatch):
    """NfoPlanner marks video without NFO as blocked."""
    import sqlite3
    from nfo_services import NfoPlanner
    from scanner import ensure_db_schema

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))
    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)

    # Create video without NFO
    video = tmp_path / "video.mp4"
    video.touch()

    file_hash = "test_hash_456"
    conn.execute(
        "INSERT INTO scanned_files (file_hash, file_path, file_name, face_count, status) "
        "VALUES (?, ?, ?, 1, 'completed')",
        (file_hash, str(video), "video.mp4"),
    )
    conn.commit()

    planner = NfoPlanner(str(db_path))
    items = planner.build_plan([file_hash])

    assert len(items) == 1
    item = items[0]
    assert item.nfo_path is None
    assert item.can_update is False
    assert item.risk_level == "blocked"

    conn.close()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py -k "planner_builds" -v
pytest tests/test_nfo_services.py -k "planner_marks_missing" -v
```
Expected: All FAIL

**Step 3: Implement NfoPlanner (basic version)**

Add to `nfo_services.py`:

```python
import sqlite3
import os


class NfoPlanner:
    """Builds metadata change plans for NFO files."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nfo_service = NfoService()

    def build_plan(self, file_hashes: list[str]) -> list[NfoPlanItem]:
        """Build plan for given file hashes."""
        items = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for file_hash in file_hashes:
                item = self._build_item_for_video(file_hash, conn)
                if item:
                    items.append(item)
        return items

    def _build_item_for_video(
        self, file_hash: str, conn: sqlite3.Connection
    ) -> NfoPlanItem | None:
        """Build a plan item for a single video."""
        # Get video info
        row = conn.execute(
            "SELECT file_path, file_name FROM scanned_files WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if not row:
            return None

        file_path = row["file_path"]
        file_name = row["file_name"]
        file_extension = os.path.splitext(file_name)[1] if file_name else None

        # Get DB people for this video
        db_people = self._get_db_people(file_hash, conn)

        # Find NFO
        nfo_path = self.nfo_service.find_nfo_path(file_path) if file_path else None

        if nfo_path is None:
            return self._make_blocked_item(
                file_hash, file_path, file_name, file_extension, db_people,
                reason="No NFO file found"
            )

        # Read existing actors from NFO
        try:
            actors = self.nfo_service.read_actors(nfo_path)
        except NfoParseError as e:
            return self._make_blocked_item(
                file_hash, file_path, file_name, file_extension, db_people,
                nfo_path=nfo_path, reason=str(e)
            )

        # Separate indexium vs other actors
        existing_indexium = [a.name for a in actors if a.source == "indexium"]
        other_actors = [a for a in actors if a.source != "indexium"]

        # Calculate changes
        db_people_set = set(db_people)
        existing_set = set(existing_indexium)
        tags_to_add = sorted(db_people_set - existing_set, key=str.lower)
        tags_to_remove = sorted(existing_set - db_people_set, key=str.lower)
        result_people = sorted(db_people_set, key=str.lower)

        # Determine risk level
        risk_level = self._determine_risk_level(
            nfo_path, other_actors, db_people, tags_to_remove
        )

        # Build comment strings for UI compatibility
        existing_comment = f"People: {', '.join(existing_indexium)}" if existing_indexium else ""
        result_comment = f"People: {', '.join(result_people)}" if result_people else ""

        # Get NFO mtime
        try:
            nfo_mtime = os.path.getmtime(nfo_path)
        except OSError:
            nfo_mtime = None

        return NfoPlanItem(
            file_hash=file_hash,
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            nfo_path=nfo_path,
            db_people=db_people,
            existing_people=existing_indexium,
            result_people=result_people,
            tags_to_add=tags_to_add,
            tags_to_remove=tags_to_remove,
            existing_indexium_actors=existing_indexium,
            other_actors=other_actors,
            existing_comment=existing_comment,
            result_comment=result_comment,
            risk_level=risk_level,
            can_update=True,
            metadata_only_people=[a.name for a in other_actors],
            tag_count=len(result_people),
            new_tag_count=len(tags_to_add),
            file_modified_time=nfo_mtime,
        )

    def _get_db_people(self, file_hash: str, conn: sqlite3.Connection) -> list[str]:
        """Get all people tagged for this video in the DB."""
        people = set()

        # From faces table
        rows = conn.execute(
            "SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL",
            (file_hash,),
        ).fetchall()
        for row in rows:
            if row["person_name"]:
                people.add(row["person_name"])

        # From video_people table (manual tagging)
        rows = conn.execute(
            "SELECT person_name FROM video_people WHERE file_hash = ?",
            (file_hash,),
        ).fetchall()
        for row in rows:
            if row["person_name"]:
                people.add(row["person_name"])

        return sorted(people, key=str.lower)

    def _determine_risk_level(
        self,
        nfo_path: str | None,
        other_actors: list[NfoActor],
        db_people: list[str],
        tags_to_remove: list[str],
    ) -> str:
        """Determine risk level for plan item."""
        if nfo_path is None:
            return "blocked"

        # Check for collision with non-indexium actors
        other_names = {a.name.lower() for a in other_actors}
        db_names = {n.lower() for n in db_people}
        if other_names & db_names:
            return "danger"

        if tags_to_remove:
            return "warning"

        return "safe"

    def _make_blocked_item(
        self,
        file_hash: str,
        file_path: str | None,
        file_name: str | None,
        file_extension: str | None,
        db_people: list[str],
        nfo_path: str | None = None,
        reason: str = "Unknown error",
    ) -> NfoPlanItem:
        """Create a blocked plan item."""
        return NfoPlanItem(
            file_hash=file_hash,
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            nfo_path=nfo_path,
            db_people=db_people,
            existing_people=[],
            result_people=[],
            tags_to_add=[],
            tags_to_remove=[],
            existing_indexium_actors=[],
            other_actors=[],
            existing_comment="",
            result_comment="",
            risk_level="blocked",
            can_update=False,
            probe_error=reason,
            issues=[reason],
            issue_codes=["nfo_blocked"],
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "planner" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoPlanner.build_plan with basic planning"
```

---

## Phase 7: NfoWriter (Async Write Operations)

### Task 12: Implement NfoWriter Basic Structure

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing tests**

Add to `tests/test_nfo_services.py`:

```python
import time


def test_writer_start_operation_creates_record(tmp_path, monkeypatch):
    """NfoWriter.start_operation creates DB record and returns ID."""
    import sqlite3
    from nfo_services import NfoWriter, NfoPlanItem
    from scanner import ensure_db_schema

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))
    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)
    conn.close()

    writer = NfoWriter(str(db_path))

    # Create a simple plan item
    nfo = tmp_path / "video.nfo"
    shutil.copy(FIXTURES_DIR / "empty_actors.nfo", nfo)

    item = NfoPlanItem(
        file_hash="hash1",
        file_path=str(tmp_path / "video.mp4"),
        file_name="video.mp4",
        file_extension=".mp4",
        nfo_path=str(nfo),
        db_people=["Alice"],
        existing_people=[],
        result_people=["Alice"],
        tags_to_add=["Alice"],
        tags_to_remove=[],
        existing_indexium_actors=[],
        other_actors=[],
        existing_comment="",
        result_comment="People: Alice",
        risk_level="safe",
        can_update=True,
    )

    operation_id = writer.start_operation([item])
    assert operation_id > 0

    # Check DB record exists
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT * FROM metadata_operations WHERE id = ?", (operation_id,)
    ).fetchone()
    assert row is not None
    conn.close()

    # Wait for operation to complete
    time.sleep(0.5)


def test_writer_writes_nfo_actors(tmp_path, monkeypatch):
    """NfoWriter actually writes actors to NFO file."""
    import sqlite3
    from nfo_services import NfoWriter, NfoPlanItem, NfoService
    from scanner import ensure_db_schema

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))
    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)
    conn.close()

    writer = NfoWriter(str(db_path))
    nfo_service = NfoService()

    nfo = tmp_path / "video.nfo"
    shutil.copy(FIXTURES_DIR / "empty_actors.nfo", nfo)

    item = NfoPlanItem(
        file_hash="hash1",
        file_path=str(tmp_path / "video.mp4"),
        file_name="video.mp4",
        file_extension=".mp4",
        nfo_path=str(nfo),
        db_people=["Alice", "Bob"],
        existing_people=[],
        result_people=["Alice", "Bob"],
        tags_to_add=["Alice", "Bob"],
        tags_to_remove=[],
        existing_indexium_actors=[],
        other_actors=[],
        existing_comment="",
        result_comment="People: Alice, Bob",
        risk_level="safe",
        can_update=True,
    )

    operation_id = writer.start_operation([item])

    # Wait for completion
    for _ in range(20):
        status = writer.get_operation_status(operation_id)
        if status and status.get("status") in ("completed", "failed"):
            break
        time.sleep(0.1)

    # Verify NFO was written
    actors = nfo_service.read_actors(str(nfo))
    names = {a.name for a in actors if a.source == "indexium"}
    assert names == {"Alice", "Bob"}
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py -k "writer" -v
```
Expected: All FAIL

**Step 3: Implement NfoWriter**

Add to `nfo_services.py`:

```python
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class NfoWriterRuntime:
    """Runtime state for an active write operation."""

    operation_id: int
    pause_event: threading.Event
    cancel_event: threading.Event
    written_nfos: set = field(default_factory=set)


class NfoWriter:
    """Async writer for NFO metadata with pause/resume/cancel."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nfo_service = NfoService()
        self.backup_manager = NfoBackupManager()
        self._active_operations: dict[int, NfoWriterRuntime] = {}
        self._lock = threading.Lock()

    def start_operation(self, items: list[NfoPlanItem], backup: bool = True) -> int:
        """Create operation record, start background thread, return operation_id."""
        # Filter to only items that need updates
        items_to_write = [i for i in items if i.can_update and i.requires_update]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO metadata_operations (operation_type, status, file_count) "
                "VALUES ('nfo_write', 'pending', ?)",
                (len(items_to_write),),
            )
            operation_id = cursor.lastrowid

            # Insert operation items
            for item in items_to_write:
                cursor.execute(
                    """
                    INSERT INTO metadata_operation_items (
                        operation_id, file_hash, file_path, status,
                        previous_comment, new_comment, tags_added, tags_removed, nfo_path
                    ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        item.file_hash,
                        item.file_path or "",
                        item.existing_comment,
                        item.result_comment,
                        str(item.tags_to_add),
                        str(item.tags_to_remove),
                        item.nfo_path,
                    ),
                )
            conn.commit()

        # Create runtime
        runtime = NfoWriterRuntime(
            operation_id=operation_id,
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
        )
        runtime.pause_event.set()  # Not paused initially

        with self._lock:
            self._active_operations[operation_id] = runtime

        # Start background thread
        thread = threading.Thread(
            target=self._process_loop,
            args=(operation_id, items_to_write, backup),
            daemon=True,
        )
        thread.start()

        return operation_id

    def pause_operation(self, operation_id: int) -> bool:
        """Pause running operation."""
        with self._lock:
            runtime = self._active_operations.get(operation_id)
            if not runtime:
                return False
            runtime.pause_event.clear()
            self._update_operation_status(operation_id, "paused")
            return True

    def resume_operation(self, operation_id: int) -> bool:
        """Resume paused operation."""
        with self._lock:
            runtime = self._active_operations.get(operation_id)
            if not runtime:
                return False
            runtime.pause_event.set()
            self._update_operation_status(operation_id, "in_progress")
            return True

    def cancel_operation(self, operation_id: int) -> bool:
        """Cancel running operation."""
        with self._lock:
            runtime = self._active_operations.get(operation_id)
            if not runtime:
                return False
            runtime.cancel_event.set()
            runtime.pause_event.set()  # Unblock if paused
            self._update_operation_status(operation_id, "cancelling")
            return True

    def get_operation_status(self, operation_id: int) -> dict | None:
        """Get current operation status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def _process_loop(
        self, operation_id: int, items: list[NfoPlanItem], backup: bool
    ) -> None:
        """Background thread: process items with pause/cancel checks."""
        runtime = self._active_operations.get(operation_id)
        if not runtime:
            return

        self._update_operation_status(operation_id, "in_progress")

        try:
            for item in items:
                # Check for pause
                runtime.pause_event.wait()

                # Check for cancel
                if runtime.cancel_event.is_set():
                    self._update_operation_status(operation_id, "cancelled")
                    return

                # Process item
                self._write_single_item(item, operation_id, backup, runtime)

            self._update_operation_status(operation_id, "completed")
        except Exception as e:
            logger.exception("Error in write operation %d", operation_id)
            self._update_operation_status(operation_id, "failed", str(e))
        finally:
            with self._lock:
                self._active_operations.pop(operation_id, None)

    def _write_single_item(
        self,
        item: NfoPlanItem,
        operation_id: int,
        backup: bool,
        runtime: NfoWriterRuntime,
    ) -> None:
        """Write a single NFO file."""
        if not item.nfo_path:
            self._mark_item_status(operation_id, item.file_hash, "skipped")
            return

        # Check if already written (shared NFO)
        if item.nfo_path in runtime.written_nfos:
            self._mark_item_status(operation_id, item.file_hash, "success")
            self._increment_success(operation_id)
            return

        try:
            # Create backup if enabled
            if backup:
                self.backup_manager.create_backup(item.nfo_path, operation_id)

            # Write actors
            self.nfo_service.write_actors(item.nfo_path, item.result_people)

            # Mark as written
            runtime.written_nfos.add(item.nfo_path)
            self._mark_item_status(operation_id, item.file_hash, "success")
            self._increment_success(operation_id)

        except Exception as e:
            logger.exception("Failed to write NFO %s", item.nfo_path)
            self._mark_item_status(operation_id, item.file_hash, "failed", str(e))
            self._increment_failure(operation_id)

    def _update_operation_status(
        self, operation_id: int, status: str, error: str | None = None
    ) -> None:
        """Update operation status in DB."""
        with sqlite3.connect(self.db_path) as conn:
            if error:
                conn.execute(
                    "UPDATE metadata_operations SET status = ?, error_message = ? WHERE id = ?",
                    (status, error, operation_id),
                )
            else:
                conn.execute(
                    "UPDATE metadata_operations SET status = ? WHERE id = ?",
                    (status, operation_id),
                )
            conn.commit()

    def _mark_item_status(
        self, operation_id: int, file_hash: str, status: str, error: str | None = None
    ) -> None:
        """Mark individual item status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE metadata_operation_items SET status = ?, error_message = ?, "
                "processed_at = CURRENT_TIMESTAMP WHERE operation_id = ? AND file_hash = ?",
                (status, error, operation_id, file_hash),
            )
            conn.commit()

    def _increment_success(self, operation_id: int) -> None:
        """Increment success count."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE metadata_operations SET success_count = success_count + 1 WHERE id = ?",
                (operation_id,),
            )
            conn.commit()

    def _increment_failure(self, operation_id: int) -> None:
        """Increment failure count."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE metadata_operations SET failure_count = failure_count + 1 WHERE id = ?",
                (operation_id,),
            )
            conn.commit()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "writer" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoWriter with async operations"
```

---

## Phase 8: NfoHistoryService

### Task 13: Implement NfoHistoryService

**Files:**
- Modify: `nfo_services.py`
- Modify: `tests/test_nfo_services.py`

**Step 1: Write failing tests**

Add to `tests/test_nfo_services.py`:

```python
def test_history_service_list_operations(tmp_path, monkeypatch):
    """NfoHistoryService.list_operations returns operation list."""
    import sqlite3
    from nfo_services import NfoHistoryService
    from scanner import ensure_db_schema

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))
    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)

    # Insert test operation
    conn.execute(
        "INSERT INTO metadata_operations (operation_type, status, file_count) "
        "VALUES ('nfo_write', 'completed', 5)"
    )
    conn.commit()
    conn.close()

    history = NfoHistoryService(str(db_path))
    ops, total = history.list_operations(limit=10)

    assert total >= 1
    assert len(ops) >= 1
    assert ops[0]["status"] == "completed"


def test_history_service_rollback_operation(tmp_path, monkeypatch):
    """NfoHistoryService.rollback_operation restores from backup."""
    import sqlite3
    from nfo_services import NfoHistoryService, NfoBackupManager, NfoService
    from scanner import ensure_db_schema

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("scanner.DATABASE_FILE", str(db_path))
    conn = sqlite3.connect(str(db_path))
    ensure_db_schema(conn)

    # Create NFO and backup
    nfo = tmp_path / "video.nfo"
    original_content = '<?xml version="1.0"?><movie><title>Original</title></movie>'
    nfo.write_text(original_content)

    backup_manager = NfoBackupManager()
    backup_manager.create_backup(str(nfo), operation_id=1)

    # Modify NFO
    nfo.write_text('<?xml version="1.0"?><movie><title>Modified</title></movie>')

    # Insert operation record
    conn.execute(
        "INSERT INTO metadata_operations (id, operation_type, status, file_count) "
        "VALUES (1, 'nfo_write', 'completed', 1)"
    )
    conn.execute(
        "INSERT INTO metadata_operation_items (operation_id, file_hash, file_path, status, nfo_path) "
        "VALUES (1, 'hash1', ?, 'success', ?)",
        (str(tmp_path / "video.mp4"), str(nfo)),
    )
    conn.commit()
    conn.close()

    # Rollback
    history = NfoHistoryService(str(db_path))
    result = history.rollback_operation(1)

    assert result["success"] is True
    assert "Original" in nfo.read_text()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nfo_services.py -k "history_service" -v
```
Expected: All FAIL

**Step 3: Implement NfoHistoryService**

Add to `nfo_services.py`:

```python
class NfoHistoryService:
    """Service for querying operation history and performing rollbacks."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backup_manager = NfoBackupManager()

    def list_operations(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: str | None = None,
    ) -> tuple[list[dict], int]:
        """Return recent operations for history UI."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            where = ""
            params: list = []
            if status_filter:
                where = "WHERE status = ?"
                params.append(status_filter)

            # Get total count
            total = conn.execute(
                f"SELECT COUNT(*) FROM metadata_operations {where}",
                params,
            ).fetchone()[0]

            # Get operations
            rows = conn.execute(
                f"SELECT * FROM metadata_operations {where} "
                f"ORDER BY started_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

            return [dict(row) for row in rows], total

    def get_operation_detail(self, operation_id: int) -> dict | None:
        """Get full operation details including items."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            op_row = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if not op_row:
                return None

            items = conn.execute(
                "SELECT * FROM metadata_operation_items WHERE operation_id = ?",
                (operation_id,),
            ).fetchall()

            return {
                "operation": dict(op_row),
                "items": [dict(item) for item in items],
            }

    def rollback_operation(self, operation_id: int) -> dict:
        """Rollback operation using operation-scoped backups."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Check operation exists and is rollback-able
            op = conn.execute(
                "SELECT * FROM metadata_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()

            if not op:
                return {"success": False, "error": "Operation not found"}

            if op["status"] not in ("completed", "failed"):
                return {"success": False, "error": "Operation not in rollback-able state"}

            # Get items with nfo_path
            items = conn.execute(
                "SELECT * FROM metadata_operation_items WHERE operation_id = ? AND nfo_path IS NOT NULL",
                (operation_id,),
            ).fetchall()

            restored = 0
            failed = 0
            restored_nfos = set()

            for item in items:
                nfo_path = item["nfo_path"]
                if not nfo_path or nfo_path in restored_nfos:
                    continue

                if self.backup_manager.restore_backup(nfo_path, operation_id):
                    self.backup_manager.cleanup_backup(nfo_path, operation_id)
                    restored += 1
                    restored_nfos.add(nfo_path)
                else:
                    failed += 1

            # Update operation status
            conn.execute(
                "UPDATE metadata_operations SET status = 'rolled_back' WHERE id = ?",
                (operation_id,),
            )
            conn.execute(
                "UPDATE metadata_operation_items SET status = 'rolled_back' WHERE operation_id = ?",
                (operation_id,),
            )
            conn.commit()

            return {
                "success": True,
                "restored": restored,
                "failed": failed,
            }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nfo_services.py -k "history_service" -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add nfo_services.py tests/test_nfo_services.py
git commit -m "feat: implement NfoHistoryService with rollback support"
```

---

## Phase 9: App Integration

### Task 14: Integrate NFO Services into app.py

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Step 1: Update imports in app.py**

Find the metadata_services import section and update:

```python
# Replace:
# from metadata_services import MetadataPlanner, MetadataWriter, HistoryService

# With:
from nfo_services import NfoPlanner, NfoWriter, NfoHistoryService
```

**Step 2: Update global service instances**

Find where `metadata_planner`, `metadata_writer`, etc. are instantiated and update:

```python
# Replace old instances with:
nfo_planner = NfoPlanner(db_path=DATABASE_FILE)
nfo_writer = NfoWriter(db_path=DATABASE_FILE)
nfo_history = NfoHistoryService(db_path=DATABASE_FILE)
```

**Step 3: Update route handlers to use new services**

Update `/metadata_preview`, `/write_metadata`, etc. routes to use the new service names.

This is a larger refactoring task. The exact changes depend on the current app.py structure.

**Step 4: Run full test suite**

```bash
pytest -q
```

**Step 5: Commit**

```bash
git add app.py
git commit -m "feat: integrate NFO services into app.py"
```

---

## Phase 10: UI Updates

### Task 15: Update UI Labels

**Files:**
- Modify: `templates/metadata_preview.html`

**Step 1: Update labels**

Search and replace in `metadata_preview.html`:

- `"Writing metadata to video files"` → `"Writing metadata to NFO files"`
- `"No current comment"` → `"No current actors"`

**Step 2: Commit**

```bash
git add templates/metadata_preview.html
git commit -m "feat: update UI labels for NFO metadata"
```

---

## Final Steps

### Task 16: Run Full Test Suite and Manual Verification

**Step 1: Run all tests**

```bash
pytest -q
```
Expected: All tests pass

**Step 2: Manual verification with test_vids**

```bash
# Start the app
python app.py &

# Navigate to metadata preview and verify NFO integration works
# Use browser to test the UI

# Stop the app
pkill -f "python app.py"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete NFO metadata integration"
```

---

## Summary

This implementation plan covers:

1. **Test fixtures** - NFO XML files for testing
2. **Core infrastructure** - lxml dependency, config options
3. **NfoService** - XML read/write with encoding preservation
4. **NfoBackupManager** - Operation-scoped backups
5. **Schema migration** - New cache table, nfo_path column
6. **NfoPlanItem** - UI-compatible data class
7. **NfoPlanner** - Plan building with shared NFO handling
8. **NfoWriter** - Async writes with pause/resume/cancel
9. **NfoHistoryService** - Rollback support
10. **App integration** - Route updates
11. **UI updates** - Label changes

Each task follows TDD: write failing test → implement → verify → commit.
