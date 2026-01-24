"""NFO metadata services for Jellyfin-compatible metadata management."""

from __future__ import annotations

import os
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
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


@dataclass
class NfoPlanItem:
    """Plan item with full UI/API compatibility.

    Maintains all fields expected by metadata_preview.html and app.py serialization.
    """

    # Core identifiers
    file_hash: str
    file_path: str | None
    file_name: str | None
    file_extension: str | None

    # NFO-specific (new)
    nfo_path: str | None  # None = no NFO file, skip

    # People data
    db_people: list[str]  # People tagged in Indexium DB
    existing_people: list[str]  # All existing Indexium actors (alias for UI)
    result_people: list[str]  # Final people list after operation
    tags_to_add: list[str]  # New people to add
    tags_to_remove: list[str]  # Indexium actors to remove

    # NFO-specific internal data
    existing_indexium_actors: list[str]  # Current <actor source="indexium">
    other_actors: list[NfoActor]  # Actors from Jellyfin/TMDb (preserved)

    # Comment fields (for UI compatibility - serialize actors)
    existing_comment: str | None  # Serialized existing Indexium actors
    result_comment: str  # Serialized result actors

    # Status and risk
    risk_level: str  # "safe", "warning", "danger", "blocked"
    can_update: bool  # False if no NFO file or parse error

    # Issues tracking
    issues: list[str] = field(default_factory=list)
    issue_codes: list[str] = field(default_factory=list)
    probe_error: str | None = None  # Now used for XML parse errors

    # UI display fields
    metadata_only_people: list[str] = field(default_factory=list)  # Non-indexium actors in NFO
    will_overwrite_comment: bool = False  # True if changing existing actors
    overwrites_custom_comment: bool = False  # True if danger level
    tag_count: int = 0  # len(result_people)
    new_tag_count: int = 0  # len(tags_to_add)
    file_modified_time: float | None = None  # NFO file mtime

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


class NfoPlanner:
    """Builds change plans for NFO metadata updates."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nfo_service = NfoService()

    def build_plan(self, file_hashes: list[str]) -> list[NfoPlanItem]:
        """Build plan for given file hashes.

        Creates ONE plan item PER VIDEO (not per NFO) to preserve UI/API contract.
        """
        if not file_hashes:
            return []

        items = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for file_hash in file_hashes:
                item = self._build_item_for_hash(file_hash, conn)
                if item:
                    items.append(item)

        return items

    def _build_item_for_hash(self, file_hash: str, conn: sqlite3.Connection) -> NfoPlanItem | None:
        """Build a plan item for a single file hash."""
        # Get video info
        row = conn.execute(
            "SELECT file_hash, last_known_filepath FROM scanned_files WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()

        if not row:
            return None

        video_path = row["last_known_filepath"]
        file_name = os.path.basename(video_path) if video_path else None
        file_ext = os.path.splitext(video_path)[1] if video_path else None

        # Get people from DB (faces + video_people)
        db_people = self._get_db_people(file_hash, conn)

        # Find NFO file
        nfo_path = self.nfo_service.find_nfo_path(video_path) if video_path else None

        # Read existing actors from NFO
        existing_indexium_actors: list[str] = []
        other_actors: list[NfoActor] = []
        probe_error: str | None = None
        nfo_mtime: float | None = None

        if nfo_path:
            try:
                nfo_mtime = os.path.getmtime(nfo_path)
                all_actors = self.nfo_service.read_actors(nfo_path)
                for actor in all_actors:
                    if actor.source == "indexium":
                        existing_indexium_actors.append(actor.name)
                    else:
                        other_actors.append(actor)
            except NfoParseError as e:
                probe_error = str(e)

        # Calculate changes
        db_set = set(db_people)
        existing_set = set(existing_indexium_actors)

        tags_to_add = sorted(db_set - existing_set, key=str.lower)
        tags_to_remove = sorted(existing_set - db_set, key=str.lower)
        result_people = sorted(db_set, key=str.lower)

        # Determine risk level and can_update
        can_update = nfo_path is not None and probe_error is None
        risk_level = self._determine_risk_level(
            nfo_path, probe_error, other_actors, db_people, tags_to_remove
        )

        # Build comment strings for UI compatibility
        existing_comment = ", ".join(sorted(existing_indexium_actors, key=str.lower)) or None
        result_comment = ", ".join(result_people)

        # Non-indexium actors in NFO (for UI display)
        metadata_only_people = [a.name for a in other_actors]

        return NfoPlanItem(
            file_hash=file_hash,
            file_path=video_path,
            file_name=file_name,
            file_extension=file_ext,
            nfo_path=nfo_path,
            db_people=db_people,
            existing_people=list(existing_indexium_actors),
            result_people=result_people,
            tags_to_add=tags_to_add,
            tags_to_remove=tags_to_remove,
            existing_indexium_actors=list(existing_indexium_actors),
            other_actors=other_actors,
            existing_comment=existing_comment,
            result_comment=result_comment,
            risk_level=risk_level,
            can_update=can_update,
            probe_error=probe_error,
            metadata_only_people=metadata_only_people,
            will_overwrite_comment=bool(existing_indexium_actors and tags_to_remove),
            overwrites_custom_comment=risk_level == "danger",
            tag_count=len(result_people),
            new_tag_count=len(tags_to_add),
            file_modified_time=nfo_mtime,
        )

    def _get_db_people(self, file_hash: str, conn: sqlite3.Connection) -> list[str]:
        """Get all people tagged for this video in DB."""
        people = set()

        # From faces table
        rows = conn.execute(
            "SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL",
            (file_hash,),
        ).fetchall()
        for row in rows:
            people.add(row[0])

        # From video_people table (manual tagging)
        rows = conn.execute(
            "SELECT DISTINCT person_name FROM video_people WHERE file_hash = ?",
            (file_hash,),
        ).fetchall()
        for row in rows:
            people.add(row[0])

        return sorted(people, key=str.lower)

    def _determine_risk_level(
        self,
        nfo_path: str | None,
        probe_error: str | None,
        other_actors: list[NfoActor],
        db_people: list[str],
        tags_to_remove: list[str],
    ) -> str:
        """Determine risk level for this item."""
        if nfo_path is None:
            return "blocked"  # No NFO file

        if probe_error:
            return "blocked"  # XML parse error

        # Check for corrupted state: non-indexium actors that match DB names
        other_names = {a.name.lower() for a in other_actors}
        db_names = {n.lower() for n in db_people}
        if other_names & db_names:
            return "danger"  # Would modify non-indexium actors

        if tags_to_remove:
            return "warning"  # Removing previously tagged actors

        return "safe"  # Only adding new actors
