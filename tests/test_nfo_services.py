"""Tests for NFO metadata services."""

from __future__ import annotations

import shutil
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "nfo"


def test_nfo_parse_error_has_path_and_reason():
    """NfoParseError stores path and reason."""
    from nfo_services import NfoParseError

    err = NfoParseError("/path/to/file.nfo", "XML syntax error")
    assert err.path == "/path/to/file.nfo"
    assert err.reason == "XML syntax error"
    assert "file.nfo" in str(err)
    assert "XML syntax error" in str(err)


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
    # On case-insensitive filesystems, may return lowercase path
    assert result is not None
    assert result.lower() == str(nfo).lower()


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


# --- read_actors tests ---


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


def test_read_actors_missing_file_raises(tmp_path):
    """read_actors raises NfoParseError for missing file."""
    import pytest

    from nfo_services import NfoParseError, NfoService

    nfo = tmp_path / "nonexistent.nfo"

    service = NfoService()
    with pytest.raises(NfoParseError) as exc_info:
        service.read_actors(str(nfo))

    assert "nonexistent.nfo" in str(exc_info.value)


def test_read_actors_empty_returns_empty_list(tmp_path):
    """read_actors returns empty list when no actors."""
    from nfo_services import NfoService

    nfo = tmp_path / "test.nfo"
    shutil.copy(FIXTURES_DIR / "empty_actors.nfo", nfo)

    service = NfoService()
    actors = service.read_actors(str(nfo))

    assert actors == []


# --- write_actors tests ---


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


# --- NfoBackupManager tests ---


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


# --- Database schema tests ---


def test_nfo_actor_cache_table_exists(tmp_path, monkeypatch):
    """Verify nfo_actor_cache table is created by setup_database."""
    import sqlite3

    import scanner as scanner_module

    monkeypatch.setattr(scanner_module.config, "DATABASE_FILE", str(tmp_path / "test.db"))
    monkeypatch.setattr(scanner_module, "DATABASE_FILE", str(tmp_path / "test.db"))

    scanner_module.setup_database()

    conn = sqlite3.connect(tmp_path / "test.db")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "nfo_actor_cache" in tables

    # Verify columns
    columns = {row[1] for row in conn.execute("PRAGMA table_info(nfo_actor_cache)")}
    assert {"nfo_path", "actors_json", "nfo_mtime", "updated_at"} <= columns


def test_metadata_operation_items_has_nfo_path_column(tmp_path, monkeypatch):
    """Verify metadata_operation_items has nfo_path column."""
    import sqlite3

    import scanner as scanner_module

    monkeypatch.setattr(scanner_module.config, "DATABASE_FILE", str(tmp_path / "test.db"))
    monkeypatch.setattr(scanner_module, "DATABASE_FILE", str(tmp_path / "test.db"))

    scanner_module.setup_database()

    conn = sqlite3.connect(tmp_path / "test.db")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(metadata_operation_items)")}
    assert "nfo_path" in columns


# --- NfoPlanItem tests ---


def test_nfo_plan_item_dataclass():
    """NfoPlanItem creates with all expected fields."""
    from nfo_services import NfoPlanItem

    item = NfoPlanItem(
        file_hash="abc123",
        file_path="/videos/test.mp4",
        file_name="test.mp4",
        file_extension=".mp4",
        nfo_path="/videos/test.nfo",
        db_people=["Alice", "Bob"],
        existing_people=["Alice"],
        result_people=["Alice", "Bob"],
        tags_to_add=["Bob"],
        tags_to_remove=[],
        existing_indexium_actors=["Alice"],
        other_actors=[],
        existing_comment="Alice",
        result_comment="Alice, Bob",
        risk_level="safe",
        can_update=True,
    )

    assert item.file_hash == "abc123"
    assert item.nfo_path == "/videos/test.nfo"
    assert item.tags_to_add == ["Bob"]
    assert item.requires_update is True


def test_nfo_plan_item_requires_update_false_when_no_changes():
    """requires_update is False when no tags to add or remove."""
    from nfo_services import NfoPlanItem

    item = NfoPlanItem(
        file_hash="abc123",
        file_path="/videos/test.mp4",
        file_name="test.mp4",
        file_extension=".mp4",
        nfo_path="/videos/test.nfo",
        db_people=["Alice"],
        existing_people=["Alice"],
        result_people=["Alice"],
        tags_to_add=[],
        tags_to_remove=[],
        existing_indexium_actors=["Alice"],
        other_actors=[],
        existing_comment="Alice",
        result_comment="Alice",
        risk_level="safe",
        can_update=True,
    )

    assert item.requires_update is False


def test_nfo_plan_item_requires_update_false_when_blocked():
    """requires_update is False when can_update is False."""
    from nfo_services import NfoPlanItem

    item = NfoPlanItem(
        file_hash="abc123",
        file_path="/videos/test.mp4",
        file_name="test.mp4",
        file_extension=".mp4",
        nfo_path=None,  # No NFO file
        db_people=["Alice", "Bob"],
        existing_people=[],
        result_people=["Alice", "Bob"],
        tags_to_add=["Alice", "Bob"],
        tags_to_remove=[],
        existing_indexium_actors=[],
        other_actors=[],
        existing_comment=None,
        result_comment="Alice, Bob",
        risk_level="blocked",
        can_update=False,
    )

    assert item.requires_update is False


def test_nfo_plan_item_to_dict():
    """to_dict serializes item for API."""
    from nfo_services import NfoPlanItem

    item = NfoPlanItem(
        file_hash="abc123",
        file_path="/videos/test.mp4",
        file_name="test.mp4",
        file_extension=".mp4",
        nfo_path="/videos/test.nfo",
        db_people=["Alice"],
        existing_people=["Alice"],
        result_people=["Alice"],
        tags_to_add=[],
        tags_to_remove=[],
        existing_indexium_actors=["Alice"],
        other_actors=[],
        existing_comment="Alice",
        result_comment="Alice",
        risk_level="safe",
        can_update=True,
    )

    data = item.to_dict()
    assert data["file_hash"] == "abc123"
    assert data["requires_update"] is False
    # Internal fields removed
    assert "other_actors" not in data
    assert "existing_indexium_actors" not in data


# --- NfoPlanner tests ---


def _setup_planner_db(tmp_path, monkeypatch):
    """Helper to set up test database for planner tests."""
    import scanner as scanner_module

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(scanner_module.config, "DATABASE_FILE", db_path)
    monkeypatch.setattr(scanner_module, "DATABASE_FILE", db_path)
    scanner_module.setup_database()
    return db_path


def test_nfo_planner_build_plan_with_nfo(tmp_path, monkeypatch):
    """NfoPlanner builds plan for video with NFO file."""
    import sqlite3

    from nfo_services import NfoPlanner

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    # Create video and NFO
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "test.mp4"
    video.write_bytes(b"fake video")
    nfo = video_dir / "test.nfo"
    nfo.write_text('<?xml version="1.0"?><movie><title>Test</title></movie>')

    # Insert scanned file and face with person name
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash123", str(video)),
    )
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 1, "0,0,100,100", b"fake_encoding", "Alice"),
    )
    conn.commit()
    conn.close()

    planner = NfoPlanner(db_path)
    items = planner.build_plan(["hash123"])

    assert len(items) == 1
    item = items[0]
    assert item.file_hash == "hash123"
    assert item.nfo_path == str(nfo)
    assert "Alice" in item.db_people
    assert item.can_update is True
    assert item.risk_level == "safe"


def test_nfo_planner_build_plan_without_nfo(tmp_path, monkeypatch):
    """NfoPlanner creates default NFO path when video exists but NFO doesn't."""
    import sqlite3

    from nfo_services import NfoPlanner

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    # Create video WITHOUT NFO
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "test.mp4"
    video.write_bytes(b"fake video")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash123", str(video)),
    )
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 1, "0,0,100,100", b"fake_encoding", "Alice"),
    )
    conn.commit()
    conn.close()

    planner = NfoPlanner(db_path)
    items = planner.build_plan(["hash123"])

    assert len(items) == 1
    item = items[0]
    # nfo_path is set to default location when video exists
    assert item.nfo_path == str(video_dir / "test.nfo")
    assert item.can_update is True
    assert item.risk_level == "safe"


def test_nfo_planner_detects_tags_to_add(tmp_path, monkeypatch):
    """NfoPlanner correctly identifies new tags to add."""
    import sqlite3

    from nfo_services import NfoPlanner

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "test.mp4"
    video.write_bytes(b"fake video")
    # NFO with existing indexium actor "Alice"
    nfo = video_dir / "test.nfo"
    nfo.write_text(
        '<?xml version="1.0"?><movie><actor source="indexium"><name>Alice</name></actor></movie>'
    )

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash123", str(video)),
    )
    # DB has Alice and Bob
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 1, "0,0,100,100", b"fake_encoding", "Alice"),
    )
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 2, "0,0,100,100", b"fake_encoding2", "Bob"),
    )
    conn.commit()
    conn.close()

    planner = NfoPlanner(db_path)
    items = planner.build_plan(["hash123"])

    assert len(items) == 1
    item = items[0]
    assert "Alice" in item.existing_indexium_actors
    assert "Bob" in item.tags_to_add
    assert item.requires_update is True


def test_nfo_planner_detects_tags_to_remove(tmp_path, monkeypatch):
    """NfoPlanner detects when indexium actors need removal."""
    import sqlite3

    from nfo_services import NfoPlanner

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "test.mp4"
    video.write_bytes(b"fake video")
    # NFO has Alice and Bob as indexium actors
    nfo = video_dir / "test.nfo"
    nfo.write_text(
        '<?xml version="1.0"?><movie>'
        '<actor source="indexium"><name>Alice</name></actor>'
        '<actor source="indexium"><name>Bob</name></actor>'
        "</movie>"
    )

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash123", str(video)),
    )
    # DB only has Alice (Bob was untagged)
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 1, "0,0,100,100", b"fake_encoding", "Alice"),
    )
    conn.commit()
    conn.close()

    planner = NfoPlanner(db_path)
    items = planner.build_plan(["hash123"])

    assert len(items) == 1
    item = items[0]
    assert "Bob" in item.tags_to_remove
    assert item.risk_level == "warning"  # Removal triggers warning


# --- NfoWriter tests ---


def test_nfo_writer_start_operation(tmp_path, monkeypatch):
    """NfoWriter starts an operation and writes NFO files."""
    import sqlite3
    import time

    from nfo_services import NfoPlanner, NfoService, NfoWriter

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "test.mp4"
    video.write_bytes(b"fake video")
    nfo = video_dir / "test.nfo"
    nfo.write_text('<?xml version="1.0"?><movie><title>Test</title></movie>')

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash123", str(video)),
    )
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 1, "0,0,100,100", b"fake_encoding", "Alice"),
    )
    conn.commit()
    conn.close()

    planner = NfoPlanner(db_path)
    items = planner.build_plan(["hash123"])

    writer = NfoWriter(db_path)
    operation_id = writer.start_operation(items)

    assert operation_id > 0

    # Wait for operation to complete
    for _ in range(50):  # 5 second timeout
        status = writer.get_operation_status(operation_id)
        if status and status["status"] == "completed":
            break
        time.sleep(0.1)

    # Verify NFO was written
    service = NfoService()
    actors = service.read_actors(str(nfo))
    indexium_actors = [a for a in actors if a.source == "indexium"]
    assert len(indexium_actors) == 1
    assert indexium_actors[0].name == "Alice"


def test_nfo_writer_creates_backup(tmp_path, monkeypatch):
    """NfoWriter creates backup before writing."""
    import sqlite3
    import time

    from nfo_services import NfoPlanner, NfoWriter

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "test.mp4"
    video.write_bytes(b"fake video")
    nfo = video_dir / "test.nfo"
    original_content = '<?xml version="1.0"?><movie><title>Test</title></movie>'
    nfo.write_text(original_content)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash123", str(video)),
    )
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 1, "0,0,100,100", b"fake_encoding", "Alice"),
    )
    conn.commit()
    conn.close()

    planner = NfoPlanner(db_path)
    items = planner.build_plan(["hash123"])

    writer = NfoWriter(db_path)
    operation_id = writer.start_operation(items, backup=True)

    # Wait for operation to complete
    for _ in range(50):
        status = writer.get_operation_status(operation_id)
        if status and status["status"] == "completed":
            break
        time.sleep(0.1)

    # Verify backup exists
    backup_path = Path(str(nfo) + f".bak.{operation_id}")
    assert backup_path.exists()


def test_nfo_writer_skips_items_without_updates(tmp_path, monkeypatch):
    """NfoWriter skips items that don't need updates."""
    import sqlite3
    import time

    from nfo_services import NfoPlanner, NfoWriter

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "test.mp4"
    video.write_bytes(b"fake video")
    # NFO already has Alice as indexium actor
    nfo = video_dir / "test.nfo"
    nfo.write_text(
        '<?xml version="1.0"?><movie><actor source="indexium"><name>Alice</name></actor></movie>'
    )
    original_mtime = nfo.stat().st_mtime

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scanned_files (file_hash, last_known_filepath) VALUES (?, ?)",
        ("hash123", str(video)),
    )
    conn.execute(
        """INSERT INTO faces (file_hash, frame_number, face_location, face_encoding, person_name)
           VALUES (?, ?, ?, ?, ?)""",
        ("hash123", 1, "0,0,100,100", b"fake_encoding", "Alice"),
    )
    conn.commit()
    conn.close()

    planner = NfoPlanner(db_path)
    items = planner.build_plan(["hash123"])
    assert items[0].requires_update is False

    writer = NfoWriter(db_path)
    operation_id = writer.start_operation(items)

    # Wait for operation to complete
    for _ in range(50):
        status = writer.get_operation_status(operation_id)
        if status and status["status"] == "completed":
            break
        time.sleep(0.1)

    # File should not have been modified
    assert nfo.stat().st_mtime == original_mtime


# --- NfoHistoryService tests ---


def test_history_service_list_operations(tmp_path, monkeypatch):
    """NfoHistoryService.list_operations returns operation list."""
    import sqlite3

    from nfo_services import NfoHistoryService

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    # Insert test operation
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO metadata_operations (operation_type, status, file_count) "
        "VALUES ('nfo_write', 'completed', 5)"
    )
    conn.commit()
    conn.close()

    history = NfoHistoryService(db_path)
    ops, total = history.list_operations(limit=10)

    assert total >= 1
    assert len(ops) >= 1
    assert ops[0]["status"] == "completed"


def test_history_service_rollback_operation(tmp_path, monkeypatch):
    """NfoHistoryService.rollback_operation restores from backup."""
    import sqlite3

    from nfo_services import NfoBackupManager, NfoHistoryService

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    # Create NFO and backup
    nfo = tmp_path / "video.nfo"
    original_content = '<?xml version="1.0"?><movie><title>Original</title></movie>'
    nfo.write_text(original_content)

    backup_manager = NfoBackupManager()
    backup_manager.create_backup(str(nfo), operation_id=1)

    # Modify NFO
    nfo.write_text('<?xml version="1.0"?><movie><title>Modified</title></movie>')

    # Insert operation record
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO metadata_operations (id, operation_type, status, file_count) "
        "VALUES (1, 'nfo_write', 'completed', 1)"
    )
    conn.execute(
        "INSERT INTO metadata_operation_items (operation_id, file_hash, file_path, status, nfo_path, new_comment) "
        "VALUES (1, 'hash1', ?, 'success', ?, '')",
        (str(tmp_path / "video.mp4"), str(nfo)),
    )
    conn.commit()
    conn.close()

    # Rollback
    history = NfoHistoryService(db_path)
    result = history.rollback_operation(1)

    assert result["success"] is True
    assert "Original" in nfo.read_text()


def test_history_service_get_operation_detail(tmp_path, monkeypatch):
    """NfoHistoryService.get_operation_detail returns full details."""
    import sqlite3

    from nfo_services import NfoHistoryService

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    # Insert operation and items
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO metadata_operations (id, operation_type, status, file_count) "
        "VALUES (42, 'nfo_write', 'completed', 2)"
    )
    conn.execute(
        "INSERT INTO metadata_operation_items (operation_id, file_hash, file_path, status, new_comment) "
        "VALUES (42, 'hash1', '/video1.mp4', 'success', '')"
    )
    conn.execute(
        "INSERT INTO metadata_operation_items (operation_id, file_hash, file_path, status, new_comment) "
        "VALUES (42, 'hash2', '/video2.mp4', 'success', '')"
    )
    conn.commit()
    conn.close()

    history = NfoHistoryService(db_path)
    detail = history.get_operation_detail(42)

    assert detail is not None
    assert detail["operation"]["id"] == 42
    assert len(detail["items"]) == 2


def test_history_service_rollback_nonexistent_operation(tmp_path, monkeypatch):
    """NfoHistoryService.rollback_operation returns error for missing operation."""
    from nfo_services import NfoHistoryService

    db_path = _setup_planner_db(tmp_path, monkeypatch)

    history = NfoHistoryService(db_path)
    result = history.rollback_operation(999)

    assert result["success"] is False
    assert "not found" in result["error"].lower()
