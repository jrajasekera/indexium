"""Tests for NFO metadata services."""

from __future__ import annotations


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
