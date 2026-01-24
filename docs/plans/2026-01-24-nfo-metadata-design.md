# NFO Metadata Design

Replace video file comment-based metadata with Jellyfin-compatible NFO files.

## Key Decisions

- Use `<actor source="indexium">` tags to mark Indexium-managed people
- Only update existing NFO files (skip videos without NFO)
- Preserve non-Indexium actors, only manage our own
- Operation-scoped backups (`video.nfo.bak.{operation_id}`) with rollback support
- Clean break from old comment-based system
- New classes, retire old ones
- **Full UI/API field compatibility** - `NfoPlanItem` mirrors all fields from `PlanItem`
- **One plan item per video** - Preserves UI/API contract; shared NFO writes are coordinated at execution time
- **Clear path semantics** - `file_path` = video path (display), `nfo_path` = NFO path (operations)

## What Changes

- New module `nfo_services.py` with NFO-focused classes
- Planning phase reads from NFO files instead of video comments
- Writing phase modifies NFO XML instead of re-encoding video files
- Faster operations (text file edits vs video file copies)
- New `nfo_actor_cache` table for performance
- **Schema addition**: `nfo_path` column added to `metadata_operation_items` for reliable rollback

## What Stays the Same

- Scanner is unaffected (already ignores non-video files)
- Web UI unchanged (all existing fields preserved)
- Operation history tracking pattern preserved
- Async writer with pause/resume/cancel
- All four risk levels (safe/warning/danger/blocked)

## New Module: `nfo_services.py`

### Data Classes

```python
@dataclass
class NfoActor:
    """Actor data extracted from NFO file.

    Note: raw_element is NOT cached. When writing, we always read the NFO
    fresh to get current raw elements, ensuring unknown children are preserved.
    Cache only stores the extractable fields (name, source, role, type, thumb).
    """
    name: str
    source: str | None = None      # "indexium" for our actors
    role: str | None = None
    type: str | None = None
    thumb: str | None = None
    raw_element: Element | None = field(default=None, repr=False)  # NOT cached

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
    nfo_path: str | None              # None = no NFO file, skip

    # People data
    db_people: list[str]              # People tagged in Indexium DB
    existing_people: list[str]        # All existing Indexium actors (alias for UI)
    result_people: list[str]          # Final people list after operation
    tags_to_add: list[str]            # New people to add
    tags_to_remove: list[str]         # Indexium actors to remove

    # NFO-specific internal data
    existing_indexium_actors: list[str]  # Current <actor source="indexium">
    other_actors: list[NfoActor]      # Actors from Jellyfin/TMDb (preserved)

    # Comment fields (for UI compatibility - serialize actors)
    existing_comment: str | None      # Serialized existing Indexium actors
    result_comment: str               # Serialized result actors

    # Status and risk
    risk_level: str                   # "safe", "warning", "danger", "blocked"
    can_update: bool                  # False if no NFO file or parse error

    # Issues tracking
    issues: list[str] = field(default_factory=list)
    issue_codes: list[str] = field(default_factory=list)
    probe_error: str | None = None    # Now used for XML parse errors

    # UI display fields
    metadata_only_people: list[str] = field(default_factory=list)  # Non-indexium actors in NFO
    will_overwrite_comment: bool = False   # True if changing existing actors
    overwrites_custom_comment: bool = False  # True if danger level
    tag_count: int = 0                # len(result_people)
    new_tag_count: int = 0            # len(tags_to_add)
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
```

### NfoService

Core class for reading and writing NFO files:

```python
class NfoParseError(Exception):
    """Raised when NFO file cannot be parsed."""
    pass

class NfoService:
    def find_nfo_path(self, video_path: str) -> str | None:
        """Find NFO file for a video, checking multiple naming conventions.

        Checks in order (returns first match):
        1. <video_name>.nfo / .NFO (video-specific, preferred)
        2. movie.nfo / Movie.nfo (shared, may apply to multiple videos)

        Note: Caller must handle shared NFO deduplication when movie.nfo
        is returned for multiple videos in the same directory.
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

    def write_actors(
        self,
        nfo_path: str,
        indexium_actors: list[str],
        preserve_existing: bool = True
    ) -> None:
        """Update NFO: keep non-indexium actors, replace indexium actors.

        Uses lxml with:
        - Preserved original encoding/BOM
        - Minimal formatting changes (remove_blank_text=False)
        - In-place element updates where possible

        Raises:
            NfoParseError: If XML is malformed
        """

    def get_indexium_actors(self, nfo_path: str) -> list[str]:
        """Convenience: return only names where source='indexium'."""
```

### NfoBackupManager

Operation-scoped backup/restore:

```python
class NfoBackupManager:
    def create_backup(self, nfo_path: str, operation_id: int) -> str:
        """Copy to <name>.nfo.bak.<operation_id>, return backup path."""

    def restore_backup(self, nfo_path: str, operation_id: int) -> bool:
        """Restore from operation-specific backup if exists."""

    def cleanup_backup(self, nfo_path: str, operation_id: int) -> None:
        """Remove operation-specific backup file."""

    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Remove backup files older than max_age_days. Returns count removed."""

    def find_backup_path(self, nfo_path: str, operation_id: int) -> str:
        """Return expected backup path for given NFO and operation."""
        return f"{nfo_path}.bak.{operation_id}"
```

### NfoPlanner

Builds change plans preserving 1:1 video-to-item mapping:

```python
class NfoPlanner:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nfo_service = NfoService()
        self._cache_table_ready = False

    def build_plan(self, file_hashes: list[str]) -> list[NfoPlanItem]:
        """Build plan for given file hashes.

        Creates ONE plan item PER VIDEO (not per NFO) to preserve UI/API contract.
        The file_hash → plan_item mapping must be 1:1 for selection/editing to work.

        For shared NFOs (multiple videos → same movie.nfo):
        - Each video gets its own plan item
        - Each item's db_people is MERGED from all videos sharing that NFO
        - Each item shows the same result_people (union of all videos)
        - Write coordination happens at execution time, not planning time
        """

    def _collect_shared_nfo_people(
        self,
        file_hashes: list[str],
        conn: sqlite3.Connection
    ) -> dict[str, list[str]]:
        """For each nfo_path, collect merged db_people from all videos sharing it.

        Returns: {nfo_path: [merged_people_list]}

        This allows each video's plan item to show the full merged people list
        while maintaining 1:1 video-to-item mapping.
        """

    def _build_item_for_video(
        self,
        file_hash: str,
        video_path: str,
        nfo_path: str | None,
        merged_people: list[str],
        conn: sqlite3.Connection
    ) -> NfoPlanItem:
        """Build a plan item for a single video.

        - file_hash: This video's hash (used for UI/API keying)
        - video_path: This video's path (used for display)
        - nfo_path: The resolved NFO path (may be shared with other videos)
        - merged_people: Union of db_people from all videos sharing this NFO
        """

    def _ensure_cache_table(self, conn: sqlite3.Connection) -> None:
        """Create nfo_actor_cache table if missing."""

    def _get_cached_actors(
        self,
        conn: sqlite3.Connection,
        nfo_path: str
    ) -> tuple[list[NfoActor], float] | None:
        """Return (cached actors, mtime) if NFO hasn't changed, else None.

        Cache is keyed by nfo_path (not file_hash).
        Cache stores extractable fields only (name, source, role, type, thumb).
        raw_element is NOT cached - always read fresh when writing.
        """

    def _update_cache(
        self,
        conn: sqlite3.Connection,
        nfo_path: str,
        actors: list[NfoActor],
        nfo_mtime: float
    ) -> None:
        """Update actor cache for NFO path.

        Serializes actors WITHOUT raw_element to JSON.
        """
```

### Risk Level Determination

```python
def _determine_risk_level(self, item: NfoPlanItem) -> str:
    if item.nfo_path is None:
        return "blocked"  # No NFO file

    if item.probe_error:
        return "blocked"  # XML parse error

    # Check for corrupted state: actors without source="indexium" that match DB names
    # (shouldn't happen normally, but defensive)
    other_names = {a.name.lower() for a in item.other_actors}
    db_names = {n.lower() for n in item.db_people}
    if other_names & db_names:
        return "danger"  # Would modify non-indexium actors

    if item.tags_to_remove:
        return "warning"  # Removing previously tagged actors

    return "safe"  # Only adding new actors
```

### NfoWriter

Async writer with pause/resume/cancel and shared NFO coordination:

```python
@dataclass
class NfoWriterRuntime:
    operation_id: int
    pause_event: threading.Event
    cancel_event: threading.Event
    written_nfos: set[str] = field(default_factory=set)  # Track written NFO paths

class NfoWriter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nfo_service = NfoService()
        self.backup_manager = NfoBackupManager()
        self._active_operations: dict[int, NfoWriterRuntime] = {}
        self._lock = threading.Lock()

    def start_operation(
        self,
        items: list[NfoPlanItem],
        backup: bool = True
    ) -> int:
        """Create operation record, start background thread, return operation_id.

        Stores nfo_path in metadata_operation_items for rollback reliability.
        """

    def pause_operation(self, operation_id: int) -> bool:
        """Pause running operation."""

    def resume_operation(self, operation_id: int) -> bool:
        """Resume paused operation."""

    def cancel_operation(self, operation_id: int) -> bool:
        """Cancel running operation."""

    def get_operation_status(self, operation_id: int) -> dict | None:
        """Get current operation status with item details."""

    def _process_loop(self, operation_id: int, backup: bool) -> None:
        """Background thread: process items with pause/cancel checks.

        IMPORTANT: Tracks written NFO paths to avoid duplicate writes.
        Multiple videos may share the same NFO - we only write it once.
        """

    def _write_single_item(
        self,
        item: NfoPlanItem,
        operation_id: int,
        backup: bool,
        runtime: NfoWriterRuntime
    ) -> str:
        """Write a single NFO file, coordinating shared NFO writes.

        Returns: "success", "skipped" (already written), or "failed"

        Shared NFO coordination:
        1. Check if nfo_path already in runtime.written_nfos
        2. If yes: mark item as "success" (inherits the write) without re-writing
        3. If no: write NFO, add to written_nfos, mark item as "success"

        Write steps (when not skipped):
        1. Read current NFO fresh (to get raw_element for unknown children)
        2. Create backup if enabled
        3. Update actors (remove old indexium, add new)
        4. Write NFO preserving formatting
        5. Add nfo_path to runtime.written_nfos
        """
```

### NfoHistoryService

Tracks operations and handles rollback:

```python
class NfoHistoryService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backup_manager = NfoBackupManager()

    def list_operations(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: str | None = None
    ) -> tuple[list[dict], int]:
        """Return recent operations for history UI."""

    def get_operation_detail(self, operation_id: int) -> dict | None:
        """Get full operation details including items."""

    def rollback_operation(self, operation_id: int) -> dict:
        """Rollback operation using operation-scoped backups.

        Uses nfo_path stored in metadata_operation_items (not recomputed
        from video path) to ensure correct backup is restored even if
        NFO files have moved.

        For each item:
        1. Read nfo_path from metadata_operation_items
        2. Find .nfo.bak.{operation_id} file
        3. Restore original NFO
        4. Remove backup
        5. Update item status to 'rolled_back'
        """

    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Remove old backup files."""
```

## Database Changes

### New Table: `nfo_actor_cache`

```sql
CREATE TABLE IF NOT EXISTS nfo_actor_cache (
    nfo_path TEXT PRIMARY KEY,     -- Keyed by NFO path, not file_hash
    actors_json TEXT,              -- JSON array of NfoActor objects (without raw_element)
    nfo_mtime REAL,                -- NFO file modification time
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Notes**:
- Cache is keyed by `nfo_path` (not `file_hash`) because multiple videos can share a single `movie.nfo` file
- `actors_json` stores extractable fields only: `name`, `source`, `role`, `type`, `thumb`
- `raw_element` is NOT cached - we always read the NFO fresh when writing to preserve unknown children/attributes

### Schema Migration: Add `nfo_path` Column

```sql
-- Add nfo_path column to metadata_operation_items for reliable rollback
ALTER TABLE metadata_operation_items ADD COLUMN nfo_path TEXT;

-- Index for efficient lookups during rollback
CREATE INDEX IF NOT EXISTS idx_metadata_items_nfo_path
    ON metadata_operation_items (nfo_path);
```

**Migration strategy**:
- `scanner.py` `ensure_db_schema()` will add the column if missing (SQLite `ALTER TABLE ADD COLUMN` is safe)
- Existing rows will have `nfo_path = NULL` (old operations can't be rolled back via NFO, but they used ffmpeg anyway)

### Table Creation in scanner.py

Add to `ensure_db_schema()`:

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

### Existing Tables: Semantic Changes

The following columns keep their names but change semantics:

- `metadata_operation_items.previous_comment` - Now stores serialized previous Indexium actors
- `metadata_operation_items.new_comment` - Now stores serialized new Indexium actors
- `metadata_history.original_comment` - Now stores full NFO actor section XML
- `metadata_history.original_metadata_json` - Now stores full NFO XML (for complete restoration)

**Path semantics clarification**:

| Column | Contains | Used For |
|--------|----------|----------|
| `metadata_operation_items.file_path` | Video file path | Display in UI, identifying the video |
| `metadata_operation_items.nfo_path` | NFO file path | Rollback operations, backup location |

Both paths are stored because:
1. `file_path` (video) is what users see and search for
2. `nfo_path` is what we actually modify and need for rollback
3. They may differ (video.mp4 → movie.nfo) and nfo_path may be shared

## Configuration

New config options in `config.py`:

```python
# NFO Metadata Settings
NFO_REMOVE_STALE_ACTORS: bool = env_bool("NFO_REMOVE_STALE_ACTORS", True)
    # When True, remove Indexium actors no longer in DB
    # When False, only add new actors, never remove

NFO_BACKUP_MAX_AGE_DAYS: int = env_int("NFO_BACKUP_MAX_AGE_DAYS", 30)
    # Days to keep old backup files before cleanup
```

## App Integration

### Flask Routes

Swap service imports, route handlers stay same shape:

```python
# In app.py

# Old imports (to be removed):
# from metadata_services import MetadataPlanner, MetadataWriter, HistoryService

# New imports:
from nfo_services import NfoPlanner, NfoWriter, NfoHistoryService

# Global instances
nfo_planner = NfoPlanner(db_path=DATABASE_FILE)
nfo_writer = NfoWriter(db_path=DATABASE_FILE)
nfo_history = NfoHistoryService(db_path=DATABASE_FILE)

# Routes use same patterns, swap underlying service calls
```

### UI Template Changes

Minor updates to `templates/metadata_preview.html`:

```javascript
// Update labels (search and replace)
"Writing metadata to video files" → "Writing metadata to NFO files"
"No current comment" → "No current actors"
"Will overwrite existing custom comment" → "Will modify actors"

// Risk level explanations (update tooltip/help text)
const riskDescriptions = {
    safe: "Only adding new actors to NFO",
    warning: "Will remove some previously tagged actors",
    danger: "May affect non-Indexium actors (review carefully)",
    blocked: "No NFO file found or XML parse error"
};

// Add removal warning display
// When item.tags_to_remove is non-empty, show prominently:
// "Will remove: Alice, Bob"
```

Show removal warnings prominently when `tags_to_remove` is non-empty. This addresses the behavioral change concern.

## XML Handling

Use `lxml` for robust XML processing with error handling:

```python
from lxml import etree

class NfoParseError(Exception):
    """Raised when NFO cannot be parsed."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse {path}: {reason}")

class NfoXmlHandler:
    @staticmethod
    def read(path: str) -> tuple[etree._Element, str | None]:
        """Read NFO file, return (root element, detected encoding).

        Raises:
            NfoParseError: If file is not valid XML
        """
        try:
            with open(path, 'rb') as f:
                raw = f.read()
        except OSError as e:
            raise NfoParseError(path, f"Cannot read file: {e}")

        # Detect BOM and encoding
        encoding = None
        if raw.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
            raw = raw[3:]
            encoding = 'utf-8-sig'

        # Try parsing with recovery mode for slightly malformed XML
        try:
            parser = etree.XMLParser(
                remove_blank_text=False,
                recover=True  # Attempt to recover from errors
            )
            root = etree.fromstring(raw, parser)

            # Check if recovery mode produced a usable result
            if root is None:
                raise NfoParseError(path, "XML recovery failed")

            return root, encoding

        except etree.XMLSyntaxError as e:
            raise NfoParseError(path, f"XML syntax error: {e}")

    @staticmethod
    def write(path: str, root: etree._Element, encoding: str | None) -> None:
        """Write NFO file, preserving original encoding.

        Minimizes formatting changes by:
        - Not pretty-printing
        - Preserving original encoding declaration
        - Preserving BOM if present
        """
        xml_bytes = etree.tostring(
            root,
            encoding=encoding or 'utf-8',
            xml_declaration=True,
            pretty_print=False  # Minimize formatting changes
        )

        with open(path, 'wb') as f:
            if encoding == 'utf-8-sig':
                f.write(b'\xef\xbb\xbf')  # Write BOM
            f.write(xml_bytes)
```

### Handling Parse Failures

When NFO parsing fails, the plan item is marked as blocked:

```python
def _build_item_for_nfo(self, nfo_path: str, ...) -> NfoPlanItem:
    try:
        actors = self.nfo_service.read_actors(nfo_path)
    except NfoParseError as e:
        return NfoPlanItem(
            # ... other fields ...
            nfo_path=nfo_path,
            can_update=False,
            risk_level="blocked",
            probe_error=str(e),
            issues=[f"Cannot parse NFO: {e.reason}"],
            issue_codes=["nfo_parse_error"],
        )
```

## Shared NFO Handling

### The Problem

When using Jellyfin's `movie.nfo` convention, multiple videos in the same directory share one NFO file:

```
/movies/
  video1.mp4  → movie.nfo
  video2.mp4  → movie.nfo (same file!)
  movie.nfo
```

This creates issues:
- UI/API keys everything by `file_hash` - deduplication breaks selection/editing
- Search/filter by video path won't find deduplicated items
- Stats say "N files" but dedup makes them "N unique NFOs"
- Duplicate writes if not coordinated

### The Solution

**Preserve 1:1 video-to-item mapping, coordinate writes at execution:**

1. **Planning**: Each video gets its own plan item (preserves UI/API contract)
2. **People merging**: Each item shows merged people from ALL videos sharing that NFO
3. **Write coordination**: Track written NFOs during execution, skip duplicates

```python
def build_plan(self, file_hashes: list[str]) -> list[NfoPlanItem]:
    # First pass: collect merged people for each NFO path
    nfo_to_people = self._collect_shared_nfo_people(file_hashes, conn)

    # Second pass: build one item per video, with merged people
    items = []
    for file_hash in file_hashes:
        video_path = self._get_video_path(file_hash, conn)
        nfo_path = self.nfo_service.find_nfo_path(video_path)
        merged_people = nfo_to_people.get(nfo_path, [])

        items.append(self._build_item_for_video(
            file_hash, video_path, nfo_path, merged_people, conn
        ))

    return items

# During write:
def _write_single_item(self, item, operation_id, backup, runtime):
    if item.nfo_path in runtime.written_nfos:
        # Already written by another video sharing this NFO
        return "success"  # Inherit the write

    # Actually write the NFO
    self._do_write(item, operation_id, backup)
    runtime.written_nfos.add(item.nfo_path)
    return "success"
```

**Benefits**:
- UI/API selection and editing work correctly (1:1 file_hash mapping)
- Search/filter finds all videos by their original paths
- Stats show actual file count (not deduplicated)
- Duplicate writes are prevented at execution time
- All videos sharing an NFO show the same merged result_people

## Assumptions and Risks

### Source Attribute Durability

The design assumes Jellyfin preserves custom attributes on `<actor>` elements. This was verified by manual testing. If Jellyfin changes this behavior in a future version:

- **Detection**: Indexium actors would become indistinguishable from other actors
- **Fallback**: Could switch to name-based tracking (compare DB names vs NFO names)
- **Mitigation**: Monitor Jellyfin release notes; add a config option to use name-based mode

### Behavioral Change: Actor Removal

The new system removes Indexium actors that are no longer in the DB by default. This differs from the old comment-based system which only merged. Users should be aware:

- Configure `NFO_REMOVE_STALE_ACTORS=false` to preserve old behavior
- UI prominently shows "Will remove: X, Y" when removals are planned
- Risk level is "warning" (not "safe") when removals are planned
- Rollback is available if removals were unintended

### XML Robustness

Many NFO files in the wild are not well-formed XML. The design handles this by:

- Using `lxml` with `recover=True` for best-effort parsing
- Marking unparseable NFOs as "blocked" with clear error messages
- Never modifying files we can't reliably parse
- Preserving original encoding and BOM to minimize churn

## Testing Strategy

### New Test File: `tests/test_nfo_services.py`

```python
# NfoService tests
def test_find_nfo_path_exact_match():
def test_find_nfo_path_case_insensitive():
def test_find_nfo_path_movie_nfo_fallback():
def test_find_nfo_path_missing_returns_none():
def test_read_actors_parses_all_actors():
def test_read_actors_captures_source_attribute():
def test_read_actors_preserves_full_structure():
def test_read_actors_malformed_xml_raises():
def test_read_actors_recovers_minor_errors():
def test_write_actors_preserves_non_indexium_actors():
def test_write_actors_replaces_indexium_actors():
def test_write_actors_preserves_encoding():
def test_write_actors_preserves_bom():

# NfoBackupManager tests
def test_create_backup_uses_operation_id():
def test_restore_backup_finds_correct_operation():
def test_restore_backup_missing_returns_false():
def test_cleanup_old_backups():

# NfoPlanner tests
def test_plan_identifies_actors_to_add():
def test_plan_identifies_actors_to_remove():
def test_plan_marks_missing_nfo_as_blocked():
def test_plan_marks_malformed_nfo_as_blocked():
def test_plan_uses_cache_when_valid():
def test_plan_risk_level_safe():
def test_plan_risk_level_warning():
def test_plan_risk_level_danger():
def test_plan_risk_level_blocked():
def test_plan_preserves_one_item_per_video():
def test_plan_merges_people_for_shared_nfo():
def test_plan_shared_nfo_all_items_have_same_result():
def test_plan_item_has_all_ui_fields():

# NfoWriter tests
def test_start_operation_creates_record():
def test_start_operation_stores_nfo_path():
def test_pause_resume_operation():
def test_cancel_operation():
def test_write_creates_backup():
def test_write_updates_nfo_actors():
def test_write_skips_already_written_nfo():
def test_write_shared_nfo_only_writes_once():
def test_write_reads_fresh_nfo_for_raw_elements():

# NfoHistoryService tests
def test_rollback_uses_stored_nfo_path():
def test_rollback_restores_correct_backup():
def test_rollback_cleans_up_backup_file():
def test_list_operations_pagination():

# Schema migration tests
def test_nfo_actor_cache_table_created():
def test_nfo_path_column_added_to_operation_items():
```

### Test Fixtures: `tests/fixtures/nfo/`

```
tests/fixtures/nfo/
├── empty_actors.nfo          # NFO with no actors
├── existing_actors.nfo       # NFO with TMDb actors
├── indexium_actors.nfo       # NFO with source="indexium" actors
├── mixed_actors.nfo          # Both Indexium and external actors
├── utf8_bom.nfo              # UTF-8 with BOM
├── complex_actors.nfo        # Actors with role, thumb, etc.
├── movie.nfo                 # Jellyfin movie.nfo naming
├── malformed.nfo             # Invalid XML for error handling tests
└── recoverable.nfo           # Slightly malformed but recoverable
```

### Modified Test Files

- `tests/test_app.py` - Update metadata route tests to use NFO mocks
- `tests/test_e2e.py` - Add NFO files to test fixtures, verify NFO output
- `tests/test_e2e_ui.py` - Update to check NFO-specific UI elements
- `tests/test_scanner.py` - Add tests for schema migration (nfo_actor_cache, nfo_path column)

### Integration Test Updates

Extend `e2e_test.py`:

1. Create test videos with matching NFO files
2. Run scanner to detect faces
3. Tag people via DB
4. Execute metadata write
5. Parse NFO and verify `<actor source="indexium">` entries
6. Verify non-Indexium actors preserved
7. Rollback and verify original NFO restored
8. Verify backup file cleaned up after rollback
9. Test shared movie.nfo scenario:
   - Multiple videos in same directory share movie.nfo
   - Plan shows one item per video (not deduplicated)
   - Each item shows merged people from all videos
   - Only one NFO write occurs (coordinated)
   - All items marked success after write
10. Test malformed NFO handling (blocked, not crashed)
11. Test cache doesn't store raw_element (verify by checking JSON)
