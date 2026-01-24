# NFO Metadata Design

Replace video file comment-based metadata with Jellyfin-compatible NFO files.

## Key Decisions

- Use `<actor source="indexium">` tags to mark Indexium-managed people
- Only update existing NFO files (skip videos without NFO)
- Preserve non-Indexium actors, only manage our own
- Git-style backup (`.nfo.bak`) with rollback support
- Clean break from old comment-based system
- New classes, retire old ones

## What Changes

- `metadata_services.py` - Old classes deprecated, new NFO-focused classes added
- Planning phase reads from NFO files instead of video comments via ffprobe
- Writing phase modifies NFO XML instead of re-encoding video files
- Faster operations (text file edits vs video file copies)

## What Stays the Same

- Scanner is unaffected (already ignores non-video files)
- Database schema unchanged (still tracks people by file hash)
- Web UI unchanged (still displays plans and triggers writes)
- Operation history tracking pattern preserved

## New Classes

### `NfoService` (new module: `nfo_services.py`)

Core class for reading and writing NFO files:

```python
@dataclass
class NfoActor:
    name: str
    source: str | None = None  # "indexium" for our actors

class NfoService:
    def find_nfo_path(self, video_path: str) -> str | None
        # Returns path to <video_name>.nfo if exists, None otherwise

    def read_actors(self, nfo_path: str) -> list[NfoActor]
        # Parse XML, return all actors with their source attribute

    def write_actors(self, nfo_path: str,
                     indexium_actors: list[str],
                     preserve_existing: bool = True) -> None
        # Update NFO: keep non-indexium actors, replace indexium actors

    def get_indexium_actors(self, nfo_path: str) -> list[str]
        # Convenience: return only names where source="indexium"
```

### `NfoBackupManager`

Simple backup/restore:

```python
class NfoBackupManager:
    def create_backup(self, nfo_path: str) -> str
        # Copy to <name>.nfo.bak, return backup path

    def restore_backup(self, nfo_path: str) -> bool
        # Restore from .nfo.bak if exists

    def cleanup_backup(self, nfo_path: str) -> None
        # Remove .nfo.bak file
```

## Planning & Writing Workflow

### `NfoPlanner`

Builds change plans by comparing DB tags vs NFO actors:

```python
@dataclass
class NfoPlanItem:
    file_hash: str
    video_path: str
    nfo_path: str | None          # None = no NFO file, skip
    db_people: list[str]          # People tagged in Indexium DB
    existing_indexium_actors: list[str]  # Current <actor source="indexium">
    other_actors: list[str]       # Actors from Jellyfin/TMDb (preserved)
    actors_to_add: list[str]      # New people to add
    actors_to_remove: list[str]   # Indexium actors no longer in DB
    can_update: bool              # False if no NFO file
    risk_level: str               # "safe", "warning", "blocked"

class NfoPlanner:
    def build_plan(self, file_hashes: list[str]) -> list[NfoPlanItem]
        # For each hash: lookup video path, find NFO, compare actors
```

### `NfoWriter`

Executes the plan:

```python
class NfoWriter:
    def execute(self, plan_items: list[NfoPlanItem],
                backup: bool = True) -> OperationResult
        # For each item:
        #   1. Create .nfo.bak if backup=True
        #   2. Remove existing <actor source="indexium"> elements
        #   3. Add new <actor source="indexium"> for each db_person
        #   4. Write updated XML
```

### Risk Levels

- `safe`: NFO exists, changes needed
- `blocked`: No NFO file found
- (No "danger" level - we only touch our own actors)

## Rollback & History

### Operation Tracking

Reuse existing `metadata_operations` table pattern:

```python
@dataclass
class NfoOperation:
    id: str
    status: str  # "pending", "in_progress", "completed", "rolled_back"
    file_count: int
    success_count: int
    failure_count: int
    created_at: datetime
    completed_at: datetime | None
```

### `NfoHistoryService`

Tracks operations and handles rollback:

```python
class NfoHistoryService:
    def record_operation(self, plan_items: list[NfoPlanItem]) -> str
        # Create operation record, return operation_id

    def record_item_success(self, operation_id: str, nfo_path: str) -> None
        # Mark item complete, store backup path reference

    def rollback_operation(self, operation_id: str) -> RollbackResult
        # For each item in operation:
        #   1. Find .nfo.bak file
        #   2. Restore original NFO
        #   3. Remove .nfo.bak
        #   4. Update status to "rolled_back"

    def list_operations(self, limit: int = 20) -> list[NfoOperation]
        # Return recent operations for history UI
```

### Backup Lifecycle

1. Before write: copy `video.nfo` → `video.nfo.bak`
2. On rollback: restore `video.nfo.bak` → `video.nfo`, delete backup
3. On cleanup (optional): remove old `.nfo.bak` files after N days

## App Integration

### Flask Routes

Minimal changes, swap underlying services:

```python
# In app.py - replace imports and service instantiation

# Old:
# from metadata_services import MetadataPlanner, MetadataWriter, HistoryService

# New:
from nfo_services import NfoPlanner, NfoWriter, NfoHistoryService

# Route handlers stay largely the same shape:
@app.route("/metadata/plan", methods=["POST"])
def create_metadata_plan():
    planner = NfoPlanner(db_path=DATABASE_FILE)
    plan = planner.build_plan(file_hashes)
    # Return plan JSON (same structure, different field names)

@app.route("/metadata/write", methods=["POST"])
def execute_metadata_write():
    writer = NfoWriter(db_path=DATABASE_FILE)
    result = writer.execute(plan_items)
    # Return operation result

@app.route("/metadata/history")
def metadata_history():
    history = NfoHistoryService(db_path=DATABASE_FILE)
    operations = history.list_operations()
    # Return operations list
```

### UI Templates

Minor label updates:
- "Writing metadata to video files" → "Writing metadata to NFO files"
- Risk level explanations updated (no more "will overwrite custom comment")
- Plan display shows NFO path instead of video path for clarity

## Testing Strategy

### Unit Tests (`tests/test_nfo_services.py`)

```python
# NfoService tests
- test_find_nfo_path_exists()
- test_find_nfo_path_missing_returns_none()
- test_read_actors_parses_all_actors()
- test_read_actors_captures_source_attribute()
- test_write_actors_preserves_non_indexium_actors()
- test_write_actors_replaces_indexium_actors()
- test_write_actors_handles_empty_nfo()

# NfoBackupManager tests
- test_create_backup_copies_file()
- test_restore_backup_overwrites_original()
- test_restore_backup_missing_returns_false()

# NfoPlanner tests
- test_plan_identifies_actors_to_add()
- test_plan_identifies_actors_to_remove()
- test_plan_marks_missing_nfo_as_blocked()

# NfoHistoryService tests
- test_rollback_restores_original_nfo()
- test_rollback_cleans_up_backup_file()
```

### Integration Test

Extend `e2e_test.py`:
1. Create test videos with matching NFO files
2. Run scanner to detect faces
3. Tag people via UI
4. Execute metadata write
5. Verify NFO contains `<actor source="indexium">` entries
6. Rollback and verify original NFO restored

### Test Fixtures

Sample NFO files with various actor configurations (empty, with existing actors, with indexium actors already present).
