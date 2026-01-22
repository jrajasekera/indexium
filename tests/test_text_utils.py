from __future__ import annotations

from text_utils import calculate_top_text_fragments


def test_top_fragments_zero_top_n() -> None:
    """Pass top_n=0, expect empty list."""
    entries = [("hello world", 1), ("testing text", 2)]
    result = calculate_top_text_fragments(entries, top_n=0)
    assert result == []


def test_top_fragments_negative_top_n() -> None:
    """Pass negative top_n, expect empty list."""
    entries = [("hello world", 1)]
    result = calculate_top_text_fragments(entries, top_n=-5)
    assert result == []


def test_top_fragments_empty_text_entries() -> None:
    """Entry with empty raw_text string should be skipped."""
    entries = [("", 1), ("valid text", 2)]
    result = calculate_top_text_fragments(entries, top_n=5)
    # Should only process "valid text", not the empty string
    assert len(result) > 0
    # Ensure empty string didn't contribute
    for item in result:
        assert item["substring"] != ""


def test_top_fragments_none_text_entry() -> None:
    """Entry with None weight should be handled."""
    entries: list[tuple[str, int | None]] = [("hello world", None)]
    result = calculate_top_text_fragments(entries, top_n=5)
    # None weight should be treated as 0, then converted to 1
    assert len(result) > 0


def test_top_fragments_zero_weight() -> None:
    """Entry with weight=0 should be treated as weight=1."""
    entries = [("hello", 0)]
    result = calculate_top_text_fragments(entries, top_n=5)
    # Zero weight is converted to 1, so fragments should still be generated
    assert len(result) > 0
    # Check that count is at least 1
    assert all(item["count"] >= 1 for item in result)


def test_top_fragments_text_too_short() -> None:
    """Text shorter than min_length should be skipped."""
    # Default MIN_FRAGMENT_LENGTH is 4
    entries = [("ab", 1), ("abc", 1)]  # Both too short
    result = calculate_top_text_fragments(entries, top_n=5)
    assert result == []


def test_top_fragments_text_exactly_min_length() -> None:
    """Text exactly at min_length should be included."""
    entries = [("abcd", 1)]  # Exactly 4 chars
    result = calculate_top_text_fragments(entries, top_n=5)
    assert len(result) == 1
    assert result[0]["substring"] == "abcd"


def test_top_fragments_custom_min_length() -> None:
    """Custom min_length parameter should be respected."""
    entries = [("abcdef", 1)]
    # With min_length=6, only the full string qualifies
    result = calculate_top_text_fragments(entries, top_n=10, min_length=6)
    assert len(result) == 1
    assert result[0]["substring"] == "abcdef"


def test_top_fragments_preserves_display_case() -> None:
    """Original case should be preserved in display substring."""
    entries = [("HeLLo WoRLd", 1)]
    result = calculate_top_text_fragments(entries, top_n=100)
    # Find the full string in results
    full_match = next((r for r in result if r["lower"] == "hello world"), None)
    assert full_match is not None
    assert full_match["substring"] == "HeLLo WoRLd"


def test_top_fragments_empty_entries() -> None:
    """Empty entries iterable should return empty list."""
    result = calculate_top_text_fragments([], top_n=5)
    assert result == []


def test_top_fragments_ranking_by_count() -> None:
    """Fragments should be ranked by count descending."""
    entries = [("hello", 1), ("hello", 1), ("world", 1)]
    result = calculate_top_text_fragments(entries, top_n=10)
    # "hello" appears twice, should have higher count
    hello_idx = next(i for i, r in enumerate(result) if r["lower"] == "hello")
    world_idx = next(i for i, r in enumerate(result) if r["lower"] == "world")
    assert hello_idx < world_idx  # hello should come before world
