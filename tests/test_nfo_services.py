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
