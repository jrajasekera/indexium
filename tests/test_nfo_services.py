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
