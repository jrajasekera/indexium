"""NFO metadata services for Jellyfin-compatible metadata management."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree


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
            raise NfoParseError(nfo_path, f"Cannot read file: {e}") from e

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
            raise NfoParseError(nfo_path, f"XML syntax error: {e}") from e

    def write_actors(
        self,
        nfo_path: str,
        indexium_actors: list[str],
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

    def _write_xml(self, nfo_path: str, root: etree._Element, encoding: str | None) -> None:
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

    @staticmethod
    def _get_child_text(elem: etree._Element, tag: str) -> str | None:
        """Get text content of child element, or None."""
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None


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
