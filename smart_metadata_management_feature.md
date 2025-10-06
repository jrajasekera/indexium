# Smart Metadata Writing Planner - Feature Specification

## Executive Summary

This document outlines the design and implementation plan for enhancing the existing Smart Metadata Writing Planner feature in Indexium. The planner provides users with a comprehensive preview and control interface before writing people tags to video file metadata, preventing accidental overwrites and enabling informed decision-making.

---

## Current State Analysis

### Existing Implementation

The application already has a foundation for metadata planning:

**Backend Components:**
- `build_metadata_plan()` - Generates per-file metadata change plans
- `write_metadata()` - Executes metadata writes using ffmpeg
- `extract_people_from_comment()` - Parses existing "People:" tags

**Frontend Components:**
- `/metadata_preview` route - Displays pending changes
- `metadata_preview.html` template - Basic preview interface
- Modal trigger in base template

**Current Capabilities:**
- ✅ Detection of existing comments
- ✅ Identification of new tags to add
- ✅ Warning about comment overwrites
- ✅ Basic file selection (checkboxes)
- ✅ Differentiation between ready and blocked files

### Gaps and Limitations

1. **User Experience:**
   - No bulk selection controls beyond select all/none
   - Limited filtering or search capabilities
   - No grouping or categorization of changes
   - Minimal visual hierarchy for change severity

2. **Conflict Management:**
   - No inline editing of planned changes
   - Cannot resolve individual conflicts without leaving the planner
   - No preview of the exact metadata that will be written

3. **Safety and Recovery:**
   - No dry-run validation
   - No change history or audit trail
   - No rollback mechanism
   - Limited error reporting detail

4. **Performance:**
   - No pagination for large file sets
   - No progress indicators for long operations
   - Synchronous processing blocks the UI

---

## Feature Requirements

### Functional Requirements

#### FR1: Enhanced Preview Interface
- **FR1.1** Display comprehensive change summary with statistics
- **FR1.2** Show file-by-file breakdown of planned changes
- **FR1.3** Visualize tag additions, removals, and merges distinctly
- **FR1.4** Highlight different severity levels (safe, warning, danger)
- **FR1.5** Support pagination for large file sets (50 items per page)

#### FR2: Advanced Selection Controls
- **FR2.1** Bulk selection: All, None, by Category, by Risk Level
- **FR2.2** Smart filters: By file type, by number of tags, by conflict type
- **FR2.3** Search: By filename, by person name, by file path
- **FR2.4** Sorting: Alphabetically, by risk, by tag count, by modification time

#### FR3: Conflict Resolution
- **FR3.1** Inline display of existing vs. planned comments
- **FR3.2** Per-file "Edit Plan" option for custom tag lists
- **FR3.3** "Skip" action to postpone individual files
- **FR3.4** "Preserve Comment" option to append rather than replace

#### FR4: Safety Mechanisms
- **FR4.1** Dry-run mode that validates without writing
- **FR4.2** File integrity validation before and after writes
- **FR4.3** Automatic backup of existing metadata
- **FR4.4** Rollback capability for recent operations
- **FR4.5** Confirmation dialog for high-risk operations

#### FR5: Operation Management
- **FR5.1** Background processing for metadata writes
- **FR5.2** Real-time progress tracking with item-level status
- **FR5.3** Graceful interruption and resume capability
- **FR5.4** Detailed success/failure reporting per file

#### FR6: History and Auditing
- **FR6.1** Log all metadata write operations
- **FR6.2** Track which tags were added/removed per operation
- **FR6.3** Record timestamp and user action type
- **FR6.4** Enable review of past changes
- **FR6.5** Support filtering history by date, file, or person

### Non-Functional Requirements

#### NFR1: Performance
- Preview generation: < 2 seconds for 1000 files
- Metadata write: < 1 second per file
- UI responsiveness: No blocking operations > 200ms

#### NFR2: Usability
- Zero learning curve for basic operations
- Clear visual feedback for all actions
- Keyboard navigation support
- Mobile-responsive design

#### NFR3: Reliability
- Atomic operations (all-or-nothing per file)
- Graceful degradation on errors
- No data loss on interruption
- Automatic cleanup of temporary files

#### NFR4: Maintainability
- Modular, testable code structure
- Comprehensive error logging
- Clear separation of concerns
- Backwards compatible with existing database

---

## Architecture Design

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Browser)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Planner    │  │   Progress   │  │   History    │      │
│  │     View     │  │    Monitor   │  │    View      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                     AJAX/WebSocket                          │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│                     Flask Backend                            │
├────────────────────────────┼─────────────────────────────────┤
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐      │
│  │   Planner    │  │   Writer     │  │   History    │      │
│  │   Service    │  │   Service    │  │   Service    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
├────────────────────────────┼─────────────────────────────────┤
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐      │
│  │   Metadata   │  │   Backup     │  │   Validator  │      │
│  │   Extractor  │  │   Manager    │  │   Module     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼──────────────┐
│         │           Data Layer                │              │
├─────────┼──────────────────┼──────────────────┼──────────────┤
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐      │
│  │    SQLite    │  │  File System │  │    Logs      │      │
│  │   Database   │  │   (Videos)   │  │   (JSON)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Diagrams

#### 1. Preview Generation Flow

```
User → Click "Write Metadata"
  │
  ▼
Generate Plan
  │
  ├──→ Query Database (faces, video_people, scanned_files)
  │
  ├──→ For Each File:
  │     ├──→ Probe Existing Metadata (ffprobe)
  │     ├──→ Extract Current Tags
  │     ├──→ Merge with Database Tags
  │     ├──→ Calculate Changes
  │     └──→ Assess Risk Level
  │
  ├──→ Categorize Results:
  │     ├──→ Ready (safe, warnings)
  │     └──→ Blocked (errors, missing files)
  │
  ▼
Display Preview UI
```

#### 2. Metadata Write Flow

```
User → Select Files + Click "Write Selected"
  │
  ▼
Validation Phase
  │
  ├──→ Verify File Accessibility
  ├──→ Check Disk Space
  └──→ Validate Metadata Format
  │
  ▼
Backup Phase
  │
  └──→ For Each File:
        └──→ Extract Current Metadata → Save to metadata_history table
  │
  ▼
Write Phase (Background Worker)
  │
  └──→ For Each File:
        ├──→ Create Temp File with New Metadata
        ├──→ Verify Temp File Integrity
        ├──→ Atomic Replace Original
        ├──→ Update Progress
        └──→ Log Result
  │
  ▼
Report Phase
  │
  ├──→ Count Success/Failures
  ├──→ Generate Detailed Report
  └──→ Update UI
```

#### 3. Rollback Flow

```
User → Select Operation from History
  │
  ▼
Validate Rollback Feasibility
  │
  ├──→ Check Files Still Exist
  ├──→ Verify Backup Metadata Exists
  └──→ Confirm No Conflicting Changes
  │
  ▼
Restore Phase
  │
  └──→ For Each Affected File:
        ├──→ Read Backup Metadata
        ├──→ Write Original Metadata Back
        ├──→ Verify Restoration
        └──→ Log Rollback
  │
  ▼
Update Database
  │
  └──→ Mark Operation as Rolled Back
```

---

## Database Schema Changes

### New Tables

#### 1. `metadata_operations`
Tracks all metadata write operations for history and rollback.

```sql
CREATE TABLE metadata_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_type TEXT NOT NULL,  -- 'write', 'rollback'
    status TEXT NOT NULL,           -- 'pending', 'in_progress', 'completed', 'failed', 'rolled_back'
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    file_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    user_note TEXT
);

CREATE INDEX idx_metadata_operations_status ON metadata_operations(status);
CREATE INDEX idx_metadata_operations_started ON metadata_operations(started_at DESC);
```

#### 2. `metadata_operation_items`
Individual file changes within an operation.

```sql
CREATE TABLE metadata_operation_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL,
    file_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,              -- 'pending', 'success', 'failed', 'skipped'
    previous_comment TEXT,
    new_comment TEXT NOT NULL,
    tags_added TEXT,                   -- JSON array
    tags_removed TEXT,                 -- JSON array
    error_message TEXT,
    processed_at TIMESTAMP,
    FOREIGN KEY (operation_id) REFERENCES metadata_operations(id) ON DELETE CASCADE,
    FOREIGN KEY (file_hash) REFERENCES scanned_files(file_hash)
);

CREATE INDEX idx_metadata_items_operation ON metadata_operation_items(operation_id);
CREATE INDEX idx_metadata_items_status ON metadata_operation_items(status);
CREATE INDEX idx_metadata_items_file ON metadata_operation_items(file_hash);
```

#### 3. `metadata_history`
Backup of original metadata for rollback capability.

```sql
CREATE TABLE metadata_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_item_id INTEGER NOT NULL,
    file_hash TEXT NOT NULL,
    backup_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    original_comment TEXT,
    original_metadata_json TEXT,        -- Full metadata dump as JSON
    FOREIGN KEY (operation_item_id) REFERENCES metadata_operation_items(id) ON DELETE CASCADE,
    FOREIGN KEY (file_hash) REFERENCES scanned_files(file_hash)
);

CREATE INDEX idx_metadata_history_file ON metadata_history(file_hash);
CREATE INDEX idx_metadata_history_operation ON metadata_history(operation_item_id);
```

### Schema Migration Plan

```python
def migrate_metadata_tables():
    """
    Migration steps:
    1. Create new tables
    2. Create indexes
    3. No data migration needed (fresh feature)
    4. Update schema version
    """
```

---

## Component Specifications

### 1. Backend Services

#### 1.1 MetadataPlanner Service

**Responsibilities:**
- Generate comprehensive metadata change plans
- Categorize files by risk level
- Calculate statistics and summaries
- Support filtering and pagination

**Key Methods:**

```python
class MetadataPlanner:
    def generate_plan(
        self,
        file_hashes: List[str] = None,
        include_blocked: bool = True
    ) -> MetadataPlan:
        """Generate comprehensive metadata change plan."""
        
    def categorize_items(
        self,
        items: List[PlanItem]
    ) -> Dict[str, List[PlanItem]]:
        """Categorize plan items by risk level."""
        
    def calculate_statistics(
        self,
        plan: MetadataPlan
    ) -> PlanStatistics:
        """Calculate summary statistics for the plan."""
        
    def filter_items(
        self,
        items: List[PlanItem],
        filters: Dict[str, Any]
    ) -> List[PlanItem]:
        """Apply user-specified filters."""
```

**Data Structures:**

```python
@dataclass
class PlanItem:
    file_hash: str
    file_path: str
    file_name: str
    db_people: List[str]
    existing_people: List[str]
    result_people: List[str]
    tags_to_add: List[str]
    tags_to_remove: List[str]
    existing_comment: Optional[str]
    result_comment: str
    risk_level: str  # 'safe', 'warning', 'danger'
    can_update: bool
    issues: List[str]
    
@dataclass
class MetadataPlan:
    items: List[PlanItem]
    statistics: PlanStatistics
    categories: Dict[str, List[PlanItem]]
    
@dataclass
class PlanStatistics:
    total_files: int
    safe_count: int
    warning_count: int
    danger_count: int
    blocked_count: int
    total_tags_to_add: int
    will_overwrite_custom: int
```

#### 1.2 MetadataWriter Service

**Responsibilities:**
- Execute metadata writes asynchronously
- Maintain operation state and progress
- Handle errors gracefully
- Create backups before writing

**Key Methods:**

```python
class MetadataWriter:
    def start_operation(
        self,
        items: List[PlanItem],
        options: WriteOptions
    ) -> OperationHandle:
        """Start async metadata write operation."""
        
    def get_operation_status(
        self,
        operation_id: int
    ) -> OperationStatus:
        """Get current status of operation."""
        
    def pause_operation(self, operation_id: int) -> bool:
        """Pause an in-progress operation."""
        
    def resume_operation(self, operation_id: int) -> bool:
        """Resume a paused operation."""
        
    def cancel_operation(self, operation_id: int) -> bool:
        """Cancel an operation."""
        
    def write_single_file(
        self,
        item: PlanItem,
        create_backup: bool = True
    ) -> WriteResult:
        """Write metadata to a single file."""
```

**Background Worker:**

```python
def metadata_write_worker(
    operation_id: int,
    items: List[PlanItem],
    options: WriteOptions
):
    """
    Background worker process for metadata writes.
    Runs in separate thread/process.
    Updates database with progress.
    """
```

#### 1.3 BackupManager Service

**Responsibilities:**
- Create metadata backups before writes
- Manage backup storage
- Support rollback operations
- Clean up old backups

**Key Methods:**

```python
class BackupManager:
    def create_backup(
        self,
        file_hash: str,
        file_path: str,
        operation_item_id: int
    ) -> BackupRecord:
        """Create backup of current metadata."""
        
    def restore_backup(
        self,
        operation_item_id: int
    ) -> bool:
        """Restore metadata from backup."""
        
    def cleanup_old_backups(
        self,
        days_to_keep: int = 90
    ) -> int:
        """Remove backups older than threshold."""
```

#### 1.4 HistoryService

**Responsibilities:**
- Track all metadata operations
- Support querying operation history
- Enable rollback of operations
- Generate operation reports

**Key Methods:**

```python
class HistoryService:
    def record_operation(
        self,
        operation: MetadataOperation
    ) -> int:
        """Record new operation."""
        
    def get_operations(
        self,
        filters: Dict[str, Any],
        page: int = 1,
        per_page: int = 50
    ) -> PaginatedOperations:
        """Query operation history."""
        
    def get_operation_details(
        self,
        operation_id: int
    ) -> OperationDetails:
        """Get full details of an operation."""
        
    def rollback_operation(
        self,
        operation_id: int
    ) -> RollbackResult:
        """Rollback a completed operation."""
```

### 2. Frontend Components

#### 2.1 Enhanced Preview Interface

**Template: `metadata_preview_v2.html`**

**Sections:**

1. **Summary Header**
   - Total files pending
   - Breakdown by risk level
   - Estimated time to complete
   - Quick action buttons

2. **Filter & Search Bar**
   - Text search (filename, person name)
   - Risk level filter (All, Safe, Warning, Danger)
   - Tag count filter
   - File type filter

3. **Bulk Actions Toolbar**
   - Select All / None / Inverse
   - Select by Category buttons
   - Batch postpone
   - Export plan as JSON

4. **File List (Paginated)**
   - Expandable cards for each file
   - Visual risk indicators
   - Inline selection checkboxes
   - Quick action buttons (Edit, Skip, View)

5. **Details Panel (Per File)**
   - Current tags vs. New tags (diff view)
   - Existing comment (if any)
   - Planned comment
   - Warning/error messages
   - Edit controls

6. **Action Footer**
   - Selected count indicator
   - Dry run button
   - Write selected button (with confirmation)
   - Cancel/back button

**Key UI Elements:**

```html
<!-- Risk Level Badge -->
<span class="badge badge-{risk_level}">
    {risk_level} <!-- safe/warning/danger -->
</span>

<!-- Tag Diff Display -->
<div class="tag-diff">
    <div class="tags-current">
        <span class="tag tag-existing">Alice</span>
        <span class="tag tag-existing">Bob</span>
    </div>
    <div class="tags-arrow">→</div>
    <div class="tags-new">
        <span class="tag tag-existing">Alice</span>
        <span class="tag tag-existing">Bob</span>
        <span class="tag tag-added">Charlie</span>
    </div>
</div>

<!-- Comment Comparison -->
<div class="comment-comparison">
    <div class="comment-before">
        <label>Current:</label>
        <pre>{existing_comment}</pre>
    </div>
    <div class="comment-after">
        <label>Planned:</label>
        <pre>{result_comment}</pre>
    </div>
</div>
```

#### 2.2 Progress Monitor

**Template: `metadata_progress.html`**

**Features:**
- Real-time progress bar
- Item-by-item status list
- Success/failure counters
- Estimated time remaining
- Pause/resume controls
- Detailed error reporting

**Update Mechanism:**
- WebSocket connection for real-time updates
- Fallback to polling if WebSocket unavailable
- Update frequency: 500ms

#### 2.3 Operation History View

**Template: `metadata_history.html`**

**Features:**
- Chronological list of operations
- Filter by date range
- Search by filename or person
- Status indicators
- Expandable details
- Rollback buttons

**Details View:**
- Files affected
- Tags added/removed per file
- Success/failure status per file
- Timestamps
- User notes (if any)

### 3. API Endpoints

#### 3.1 Planning Endpoints

```python
@app.route('/api/metadata/plan', methods=['GET', 'POST'])
def get_metadata_plan():
    """
    Generate metadata change plan.
    
    Query Parameters:
    - file_hashes: Optional list of specific files
    - page: Pagination page number
    - per_page: Items per page
    - filter: JSON filter criteria
    
    Returns: MetadataPlan (JSON)
    """

@app.route('/api/metadata/plan/<file_hash>/edit', methods=['POST'])
def edit_plan_item():
    """
    Modify planned changes for a file.
    
    Body:
    - result_people: Modified list of people tags
    
    Returns: Updated PlanItem (JSON)
    """
```

#### 3.2 Write Endpoints

```python
@app.route('/api/metadata/write', methods=['POST'])
def start_metadata_write():
    """
    Start async metadata write operation.
    
    Body:
    - file_hashes: List of files to process
    - options: WriteOptions
    
    Returns: { operation_id: int }
    """

@app.route('/api/metadata/operations/<int:operation_id>', methods=['GET'])
def get_operation_status():
    """
    Get status of ongoing operation.
    
    Returns: OperationStatus (JSON)
    """

@app.route('/api/metadata/operations/<int:operation_id>/pause', methods=['POST'])
def pause_operation():
    """Pause operation."""

@app.route('/api/metadata/operations/<int:operation_id>/resume', methods=['POST'])
def resume_operation():
    """Resume operation."""

@app.route('/api/metadata/operations/<int:operation_id>/cancel', methods=['POST'])
def cancel_operation():
    """Cancel operation."""
```

#### 3.3 History Endpoints

```python
@app.route('/api/metadata/history', methods=['GET'])
def get_operation_history():
    """
    Query operation history.
    
    Query Parameters:
    - page, per_page: Pagination
    - start_date, end_date: Date filters
    - status: Filter by status
    
    Returns: PaginatedOperations (JSON)
    """

@app.route('/api/metadata/history/<int:operation_id>', methods=['GET'])
def get_operation_details():
    """
    Get detailed operation info.
    
    Returns: OperationDetails (JSON)
    """

@app.route('/api/metadata/history/<int:operation_id>/rollback', methods=['POST'])
def rollback_operation():
    """
    Rollback an operation.
    
    Returns: RollbackResult (JSON)
    """
```

---

## User Workflows

### Workflow 1: Standard Metadata Write

```
1. User clicks "Write Metadata" from any page
   ↓
2. System generates comprehensive plan
   ↓
3. User reviews preview interface
   - Sees summary statistics
   - Reviews file-by-file changes
   - Identifies any warnings/errors
   ↓
4. User applies filters/search (optional)
   - Narrows down to specific files
   - Focuses on high-risk items
   ↓
5. User makes selections
   - Uses bulk selection tools
   - Individually checks/unchecks files
   ↓
6. User clicks "Write Selected Metadata"
   ↓
7. System shows confirmation dialog
   - Final count of files
   - Summary of changes
   - Warning about irreversibility
   ↓
8. User confirms
   ↓
9. System starts background operation
   - Creates backups
   - Writes metadata
   - Updates progress in real-time
   ↓
10. Operation completes
    ↓
11. System shows summary report
    - Success count
    - Failure details
    - Links to history
```

### Workflow 2: Conflict Resolution

```
1. User reviews plan and sees warning badge on file
   ↓
2. User clicks "View Details" on the file
   ↓
3. System expands detail panel showing:
   - Current comment: "Some custom note"
   - Planned comment: "People: Alice, Bob"
   - Warning: "Will overwrite custom comment"
   ↓
4. User has options:
   a) Edit Plan → Modify tag list
   b) Skip → Postpone this file
   c) Preserve Comment → Append instead of replace
   ↓
5. User selects "Edit Plan"
   ↓
6. Inline editor appears
   - Shows current tag list
   - User removes "Bob"
   - User adds "Charlie"
   ↓
7. User saves edits
   ↓
8. System updates plan item
   - Recalculates risk level
   - Updates preview
   ↓
9. User continues with selection
```

### Workflow 3: Rollback Operation

```
1. User realizes mistake after write operation
   ↓
2. User navigates to "Metadata History"
   ↓
3. System shows list of recent operations
   ↓
4. User identifies the problematic operation
   ↓
5. User clicks "View Details"
   ↓
6. System shows:
   - All files affected
   - What changed per file
   - Current status
   ↓
7. User clicks "Rollback Operation"
   ↓
8. System validates rollback feasibility
   - Checks files still exist
   - Verifies backups available
   ↓
9. System shows confirmation:
   - "This will restore X files to their previous state"
   - Warning about any conflicts
   ↓
10. User confirms rollback
    ↓
11. System performs rollback
    - Restores original metadata
    - Updates database
    - Logs rollback action
    ↓
12. System shows rollback report
    - Files restored
    - Any failures
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goals:**
- Set up database schema
- Implement core services
- Basic API endpoints

**Tasks:**
1. Create migration for new tables
2. Implement `MetadataPlanner` service
3. Implement `BackupManager` service
4. Create basic API endpoints
5. Write unit tests

**Deliverables:**
- Working database schema
- Backend services with tests
- API endpoints functional

### Phase 2: Enhanced Preview UI (Week 3-4)

**Goals:**
- Build rich preview interface
- Implement filtering and search
- Add bulk selection tools

**Tasks:**
1. Design UI mockups
2. Create enhanced template
3. Implement JavaScript controllers
4. Add filtering logic
5. Implement pagination
6. Add accessibility features
7. User testing

**Deliverables:**
- Functional preview interface
- Filter and search working
- Responsive design complete

### Phase 3: Async Writing & Progress (Week 5-6)

**Goals:**
- Implement background writer
- Add progress monitoring
- Handle interruptions gracefully

**Tasks:**
1. Implement `MetadataWriter` service
2. Create background worker process
3. Add progress tracking
4. Implement pause/resume
5. Build progress monitor UI
6. WebSocket integration
7. Error handling and recovery

**Deliverables:**
- Async write operations working
- Real-time progress updates
- Robust error handling

### Phase 4: History & Rollback (Week 7-8)

**Goals:**
- Implement operation history
- Add rollback capability
- Create history UI

**Tasks:**
1. Implement `HistoryService`
2. Add rollback logic
3. Create history view template
4. Implement rollback UI
5. Add operation details view
6. Testing rollback scenarios

**Deliverables:**
- Complete history tracking
- Working rollback feature
- History UI functional

### Phase 5: Polish & Optimization (Week 9-10)

**Goals:**
- Performance optimization
- UI refinements
- Comprehensive testing
- Documentation

**Tasks:**
1. Performance profiling
2. Optimize database queries
3. UI/UX improvements
4. Accessibility audit
5. Comprehensive testing
6. User documentation
7. Code review

**Deliverables:**
- Production-ready feature
- Complete test coverage
- User documentation
- Deployment guide

---

## Testing Strategy

### Unit Tests

**Backend:**
- `test_metadata_planner.py`
  - Plan generation
  - Risk categorization
  - Filtering logic
  - Statistics calculation

- `test_metadata_writer.py`
  - Single file writes
  - Backup creation
  - Error handling
  - Atomic operations

- `test_backup_manager.py`
  - Backup creation
  - Restoration
  - Cleanup

- `test_history_service.py`
  - Recording operations
  - Querying history
  - Rollback logic

**Frontend:**
- Component tests for UI elements
- Filter/search logic
- Selection state management

### Integration Tests

- End-to-end write operation
- Rollback and restore
- Progress monitoring
- Concurrent operations

### Performance Tests

- Plan generation with 10k files
- Concurrent write operations
- Database query performance
- UI responsiveness

### User Acceptance Tests

1. Standard write workflow
2. Conflict resolution workflow
3. Rollback workflow
4. Filter and search
5. Bulk operations
6. Error scenarios

---

## Security Considerations

### Input Validation

- Sanitize all user inputs
- Validate file paths
- Verify file hash integrity
- Limit operation size

### File System Safety

- Atomic file operations
- Verify file permissions before write
- Use temporary files
- Clean up on failure

### Database Security

- Parameterized queries
- Transaction isolation
- Prevent SQL injection
- Regular backups

### Access Control

- Rate limiting on write operations
- Logging of all metadata changes
- User action attribution (future: multi-user)

---

## Configuration

### New Environment Variables

```bash
# Metadata operation settings
INDEXIUM_METADATA_MAX_CONCURRENT_WRITES=3
INDEXIUM_METADATA_BACKUP_RETENTION_DAYS=90
INDEXIUM_METADATA_OPERATION_TIMEOUT_SECONDS=3600
INDEXIUM_METADATA_ENABLE_WEBSOCKET=true

# UI settings
INDEXIUM_METADATA_PREVIEW_PAGE_SIZE=50
INDEXIUM_METADATA_ENABLE_INLINE_EDIT=true
```

### Config.py Additions

```python
METADATA_MAX_CONCURRENT_WRITES: int = int(
    os.environ.get("INDEXIUM_METADATA_MAX_CONCURRENT_WRITES", "3")
)
METADATA_BACKUP_RETENTION_DAYS: int = int(
    os.environ.get("INDEXIUM_METADATA_BACKUP_RETENTION_DAYS", "90")
)
METADATA_OPERATION_TIMEOUT: int = int(
    os.environ.get("INDEXIUM_METADATA_OPERATION_TIMEOUT_SECONDS", "3600")
)
METADATA_ENABLE_WEBSOCKET: bool = _str_to_bool(
    os.environ.get("INDEXIUM_METADATA_ENABLE_WEBSOCKET", "true")
)
METADATA_PREVIEW_PAGE_SIZE: int = int(
    os.environ.get("INDEXIUM_METADATA_PREVIEW_PAGE_SIZE", "50")
)
```

---

## Error Handling

### Error Categories

1. **File System Errors**
   - File not found
   - Permission denied
   - Disk full
   - File in use

2. **FFmpeg Errors**
   - Probe failure
   - Write failure
   - Corrupt file

3. **Database Errors**
   - Connection failure
   - Constraint violation
   - Transaction timeout

4. **Validation Errors**
   - Invalid metadata format
   - Exceeds size limits
   - Invalid character encoding

### Error Handling Strategy

```python
class MetadataError(Exception):
    """Base class for metadata errors."""
    
class FileAccessError(MetadataError):
    """File system access issues."""
    
class FFmpegError(MetadataError):
    """FFmpeg operation failures."""
    
class ValidationError(MetadataError):
    """Metadata validation failures."""

# Error handling pattern
try:
    result = write_metadata(file_path, metadata)
except FileAccessError as e:
    log_error(e)
    mark_file_blocked(file_hash, str(e))
except FFmpegError as e:
    log_error(e)
    retry_with_backoff(file_path, metadata)
except ValidationError as e:
    log_error(e)
    notify_user(f"Invalid metadata: {e}")
```

---

## Monitoring & Logging

### Logging Strategy

```python
# Operation start
logger.info(
    "Metadata write started",
    extra={
        "operation_id": op_id,
        "file_count": len(items),
        "user_action": "write_metadata"
    }
)

# Per-file processing
logger.debug(
    "Processing file",
    extra={
        "operation_id": op_id,
        "file_hash": file_hash,
        "file_path": file_path
    }
)

# Errors
logger.error(
    "Metadata write failed",
    extra={
        "operation_id": op_id,
        "file_hash": file_hash,
        "error": str(e)
    },
    exc_info=True
)
```

### Metrics to Track

- Operations per day
- Success rate
- Average operation duration
- Files processed per operation
- Rollback frequency
- Error types distribution

---

## Documentation Deliverables

### User Documentation

1. **Quick Start Guide**
   - How to write metadata
   - Understanding the preview
   - Resolving conflicts

2. **Feature Guide**
   - Filtering and search
   - Bulk operations
   - Rollback operations
   - History review

3. **Troubleshooting**
   - Common errors
   - Recovery procedures
   - When to rollback

### Developer Documentation

1. **Architecture Overview**
   - System design
   - Data flow
   - Component interactions

2. **API Reference**
   - Endpoint specifications
   - Request/response formats
   - Error codes

3. **Extension Guide**
   - Adding new metadata fields
   - Custom validators
   - Plugin system (future)

---

## Future Enhancements

### Short Term (3-6 months)

1. **Batch Templates**
   - Save common tag combinations
   - Quick apply to multiple files

2. **Smart Suggestions**
   - ML-based tag recommendations
   - Duplicate detection
   - Consistency checks

3. **Export/Import**
   - Export metadata as CSV
   - Import tags from external sources
   - Sync with other tools

### Long Term (6-12 months)

1. **Multi-User Support**
   - User permissions
   - Approval workflows
   - Conflict resolution

2. **Advanced Metadata**
   - Support more metadata fields
   - Custom field definitions
   - Schema validation

3. **Integration**
   - API webhooks
   - Third-party tool integration
   - Cloud storage support

---

## Success Metrics

### Feature Adoption

- % of users using preview before write
- % of operations using filters
- Rollback usage rate

### Quality Metrics

- Error rate in metadata writes
- Average operation success rate
- User-reported issues

### Performance Metrics

- Preview generation time
- Write operation throughput
- UI response time

### User Satisfaction

- User feedback scores
- Feature request themes
- Support ticket volume

---

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Data loss during write | High | Low | Atomic operations, backups, testing |
| Performance degradation | Medium | Medium | Pagination, async processing, indexing |
| Complex UI overwhelming users | Medium | Medium | Progressive disclosure, clear defaults |
| Rollback failures | High | Low | Validate before rollback, test extensively |
| Database corruption | High | Very Low | Transactions, regular backups, monitoring |

---

## Conclusion

This Smart Metadata Writing Planner enhancement transforms metadata writing from a basic batch operation into a sophisticated, user-friendly workflow with safety mechanisms, detailed preview capabilities, and powerful management tools. The phased implementation approach allows for iterative development and user feedback integration while maintaining system stability.

The architecture is designed to be:
- **Safe**: Multiple validation layers, backups, and rollback capabilities
- **Transparent**: Clear preview of all changes before execution
- **Flexible**: Advanced filtering, editing, and customization options
- **Performant**: Async operations, pagination, and optimized queries
- **Maintainable**: Modular design, comprehensive testing, clear documentation

This feature will significantly improve user confidence and control when managing video metadata, reducing errors and providing peace of mind through comprehensive safety mechanisms.
