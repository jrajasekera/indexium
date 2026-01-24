"""NFO metadata services for Jellyfin-compatible metadata management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class NfoParseError(Exception):
    """Raised when NFO file cannot be parsed."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse {path}: {reason}")


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
