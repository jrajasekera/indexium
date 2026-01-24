# NFO Metadata Design

Replace video file comment-based metadata with Jellyfin-compatible NFO files.

## Key Decisions

- Use `<actor source="indexium">` tags to mark Indexium-managed people
- Only update existing NFO files (skip videos without NFO)
- Preserve non-Indexium actors, only manage our own
- Operation-scoped backups (`video.nfo.bak.{operation_id}`) with rollback support
- Clean break from old comment-based system
- New classes, retire old ones
- Maintain API field names for UI compatibility

## What Changes

- New module `nfo_services.py` with NFO-focused classes
- Planning phase reads from NFO files instead of video comments
- Writing phase modifies NFO XML instead of re-encoding video files
- Faster operations (text file edits vs video file copies)
- New `nfo_actor_cache` table for performance

## What Stays the Same

- Scanner is unaffected (already ignores non-video files)
- Database schema field names unchanged (semantics change: "comment" fields store serialized actor data)
- Web UI largely unchanged (minor label updates)
- Operation history tracking pattern preserved
- Async writer with pause/resume/cancel
- All four risk levels (safe/warning/danger/blocked)

## New Module: `nfo_services.py`

### Data Classes

```python
@dataclass
class NfoActor:
    name: str
    source: str | None = None      # "indexium" for our actors
    role: str | None = None
    type: str | None = None
    thumb: str | None = None
    raw_element: Element | None = None  # Preserve unknown children

@dataclass
class NfoPlanItem:
    file_hash: str
    video_path: str
    nfo_path: str | None              # None = no NFO file, skip
    db_people: list[str]              # People tagged in Indexium DB
    existing_indexium_actors: list[str]  # Current <actor source="indexium">
    other_actors: list[NfoActor]      # Actors from Jellyfin/TMDb (preserved)
    actors_to_add: list[str]          # New people to add
    actors_to_remove: list[str]       # Indexium actors no longer in DB
    can_update: bool                  # False if no NFO file
    risk_level: str                   # "safe", "warning", "danger", "blocked"
    issues: list[str] = field(default_factory=list)
    issue_codes: list[str] = field(default_factory=list)

    # API compatibility fields (map to UI expectations)
    @property
    def existing_comment(self) -> str:
        """Serialized existing Indexium actors for UI display."""
        if not self.existing_indexium_actors:
            return ""
        return f"People: {', '.join(self.existing_indexium_actors)}"

    @property
    def result_comment(self) -> str:
        """Serialized result actors for UI display."""
        result = sorted(set(self.db_people), key=str.lower)
        if not result:
            return ""
        return f"People: {', '.join(result)}"
```

### NfoService

Core class for reading and writing NFO files:

```python
class NfoService:
    def find_nfo_path(self, video_path: str) -> str | None:
        """Find NFO file for a video, checking multiple naming conventions.

        Checks in order:
        1. <video_name>.nfo / .NFO
        2. movie.nfo / Movie.nfo (Jellyfin movie convention)
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
        """Parse NFO XML, return all actors with full structure preserved."""

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
```

### NfoPlanner

Builds change plans by comparing DB tags vs NFO actors:

```python
class NfoPlanner:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nfo_service = NfoService()
        self._cache_table_ready = False

    def build_plan(self, file_hashes: list[str]) -> list[NfoPlanItem]:
        """For each hash: lookup video path, find NFO, compare actors."""

    def _ensure_cache_table(self, conn: sqlite3.Connection) -> None:
        """Create nfo_actor_cache table if missing."""

    def _get_cached_actors(
        self,
        conn: sqlite3.Connection,
        file_hash: str,
        nfo_path: str
    ) -> list[NfoActor] | None:
        """Return cached actors if NFO hasn't changed, else None."""

    def _update_cache(
        self,
        conn: sqlite3.Connection,
        file_hash: str,
        actors: list[NfoActor],
        nfo_mtime: float
    ) -> None:
        """Update actor cache for file."""
```

### Risk Level Determination

```python
def _determine_risk_level(self, item: NfoPlanItem) -> str:
    if item.nfo_path is None:
        return "blocked"  # No NFO file

    # Check for corrupted state: actors without source="indexium" that match DB names
    # (shouldn't happen normally, but defensive)
    other_names = {a.name.lower() for a in item.other_actors}
    db_names = {n.lower() for n in item.db_people}
    if other_names & db_names:
        return "danger"  # Would modify non-indexium actors

    if item.actors_to_remove:
        return "warning"  # Removing previously tagged actors

    return "safe"  # Only adding new actors
```

### NfoWriter

Async writer with pause/resume/cancel:

```python
@dataclass
class NfoWriterRuntime:
    operation_id: int
    pause_event: threading.Event
    cancel_event: threading.Event

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
        """Create operation record, start background thread, return operation_id."""

    def pause_operation(self, operation_id: int) -> bool:
        """Pause running operation."""

    def resume_operation(self, operation_id: int) -> bool:
        """Resume paused operation."""

    def cancel_operation(self, operation_id: int) -> bool:
        """Cancel running operation."""

    def get_operation_status(self, operation_id: int) -> dict | None:
        """Get current operation status with item details."""

    def _process_loop(self, operation_id: int, backup: bool) -> None:
        """Background thread: process items with pause/cancel checks."""
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

        For each item:
        1. Find .nfo.bak.{operation_id} file
        2. Restore original NFO
        3. Remove backup
        4. Update item status to 'rolled_back'
        """

    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Remove old backup files."""
```

## Database Changes

### New Table: `nfo_actor_cache`

```sql
CREATE TABLE IF NOT EXISTS nfo_actor_cache (
    file_hash TEXT PRIMARY KEY,
    actors_json TEXT,      -- JSON array of NfoActor objects
    nfo_path TEXT,         -- Path to NFO file
    nfo_mtime REAL,        -- NFO file modification time
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Existing Tables: Semantic Changes

The following tables keep their schema but change semantics:

- `metadata_operation_items.previous_comment` - Now stores serialized previous Indexium actors
- `metadata_operation_items.new_comment` - Now stores serialized new Indexium actors
- `metadata_history.original_comment` - Now stores full NFO actor section XML
- `metadata_history.original_metadata_json` - Now stores full NFO XML (for complete restoration)

Note: A future migration could rename these columns for clarity, but this is not required for functionality.

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

// Risk level explanations
const riskDescriptions = {
    safe: "Only adding new actors to NFO",
    warning: "Will remove some previously tagged actors",
    danger: "May affect non-Indexium actors (review carefully)",
    blocked: "No NFO file found for this video"
};
```

Show removal warnings prominently when `actors_to_remove` is non-empty.

## XML Handling

Use `lxml` for robust XML processing:

```python
from lxml import etree

class NfoXmlHandler:
    @staticmethod
    def read(path: str) -> tuple[etree._Element, str | None]:
        """Read NFO file, return (root element, detected encoding)."""
        # Detect BOM and encoding
        with open(path, 'rb') as f:
            raw = f.read()

        encoding = None
        if raw.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
            raw = raw[3:]
            encoding = 'utf-8-sig'

        parser = etree.XMLParser(remove_blank_text=False)
        root = etree.fromstring(raw, parser)
        return root, encoding

    @staticmethod
    def write(path: str, root: etree._Element, encoding: str | None) -> None:
        """Write NFO file, preserving original encoding."""
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
- Rollback is available if removals were unintended

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
def test_plan_uses_cache_when_valid():
def test_plan_risk_level_safe():
def test_plan_risk_level_warning():
def test_plan_risk_level_danger():
def test_plan_risk_level_blocked():

# NfoWriter tests
def test_start_operation_creates_record():
def test_pause_resume_operation():
def test_cancel_operation():
def test_write_creates_backup():
def test_write_updates_nfo_actors():

# NfoHistoryService tests
def test_rollback_restores_correct_backup():
def test_rollback_cleans_up_backup_file():
def test_list_operations_pagination():
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
└── movie.nfo                 # Jellyfin movie.nfo naming
```

### Modified Test Files

- `tests/test_app.py` - Update metadata route tests to use NFO mocks
- `tests/test_e2e.py` - Add NFO files to test fixtures, verify NFO output
- `tests/test_e2e_ui.py` - Update to check NFO-specific UI elements

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
