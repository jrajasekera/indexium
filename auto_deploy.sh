#!/bin/bash

# =============================================================================
# Auto-Deploy Script for Indexium
# Polls git remote every N minutes, pulls changes, and restarts service if needed
# =============================================================================

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# -----------------------------------------------------------------------------
# Configuration (can be overridden via environment)
# -----------------------------------------------------------------------------
PROJECT_DIR="${INDEXIUM_PROJECT_DIR:-$SCRIPT_DIR}"
VENV_PATH="${INDEXIUM_VENV_PATH:-$PROJECT_DIR/.venv}"
LOG_DIR="${INDEXIUM_LOG_DIR:-$PROJECT_DIR/logs}"
DEPLOY_LOG_FILE="$LOG_DIR/auto_deploy.log"
APP_LOG_FILE="$LOG_DIR/app.log"
PID_FILE="${INDEXIUM_PID_FILE:-$HOME/.indexium.pid}"
LOCK_FILE="${INDEXIUM_LOCK_FILE:-/tmp/indexium_deploy.lock}"
LOG_BACKUP_DIR="${INDEXIUM_LOG_BACKUP_DIR:-$LOG_DIR/backup}"

# Intervals (seconds)
POLL_INTERVAL="${INDEXIUM_POLL_INTERVAL:-300}"           # 5 minutes
WATCHDOG_INTERVAL="${INDEXIUM_WATCHDOG_INTERVAL:-60}"    # 1 minute

# Log rotation
LOG_MAX_AGE_HOURS="${INDEXIUM_LOG_MAX_AGE_HOURS:-24}"
BACKUP_RETENTION_DAYS="${INDEXIUM_BACKUP_RETENTION_DAYS:-30}"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# File Locking (prevent concurrent runs)
# -----------------------------------------------------------------------------
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Another instance of auto_deploy.sh is already running. Exiting."
    exit 1
fi
# Lock acquired, will be released when script exits

# -----------------------------------------------------------------------------
# Dependency Checks
# -----------------------------------------------------------------------------
check_dependencies() {
    local missing=()
    for cmd in git flock uv; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Missing required commands: ${missing[*]}"
        echo "Please install them before running this script."
        exit 1
    fi
}

check_dependencies

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %I:%M:%S.%6N %p')
    echo "[$timestamp] [$level] $message" | tee -a "$DEPLOY_LOG_FILE"
}

log_info()  { log "INFO" "$@"; }
log_warn()  { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# -----------------------------------------------------------------------------
# Log Rotation
# -----------------------------------------------------------------------------
rotate_logs() {
    local log_files=("$APP_LOG_FILE" "$DEPLOY_LOG_FILE")
    local files_to_archive=()
    local current_time
    current_time=$(date +%s)
    local max_age_seconds=$((LOG_MAX_AGE_HOURS * 3600))

    # Check each log file by modified time
    for log_file in "${log_files[@]}"; do
        if [[ -f "$log_file" ]]; then
            local file_mtime
            # Linux uses -c %Y, macOS uses -f %m (try Linux first to avoid GNU stat output on Linux)
            file_mtime=$(stat -c %Y "$log_file" 2>/dev/null || stat -f %m "$log_file" 2>/dev/null || echo "$current_time")
            if [[ ! "$file_mtime" =~ ^[0-9]+$ ]]; then
                log_warn "Invalid mtime for $(basename "$log_file"); using current time"
                file_mtime="$current_time"
            fi

            local age=$((current_time - file_mtime))

            if [[ $age -gt $max_age_seconds ]]; then
                files_to_archive+=("$log_file")
            fi
        fi
    done

    # Archive if any files need rotation
    if [[ ${#files_to_archive[@]} -gt 0 ]]; then
        mkdir -p "$LOG_BACKUP_DIR"

        local archive_date
        archive_date=$(date +%Y-%m-%d)
        local archive_path="$LOG_BACKUP_DIR/${archive_date}.tar.gz"

        # Handle case where archive already exists (use timestamp suffix)
        if [[ -f "$archive_path" ]]; then
            archive_date=$(date +%Y-%m-%d-%H%M%S)
            archive_path="$LOG_BACKUP_DIR/${archive_date}.tar.gz"
        fi

        log_info "Rotating ${#files_to_archive[@]} log file(s) to $archive_path"

        # Create archive with just the filenames (not full paths)
        local basenames=()
        for f in "${files_to_archive[@]}"; do
            basenames+=("$(basename "$f")")
        done

        if tar -czf "$archive_path" -C "$LOG_DIR" "${basenames[@]}" 2>/dev/null; then
            # Truncate rotated files (keeps file handles valid)
            for log_file in "${files_to_archive[@]}"; do
                : > "$log_file"
                log_info "Rotated and truncated: $(basename "$log_file")"
            done
        else
            log_error "Failed to create log archive"
        fi
    fi

    # Clean up old backups
    cleanup_old_backups
}

cleanup_old_backups() {
    if [[ ! -d "$LOG_BACKUP_DIR" ]]; then
        return 0
    fi

    local cutoff_time
    # macOS uses -v, Linux uses -d
    cutoff_time=$(date -v-${BACKUP_RETENTION_DAYS}d +%s 2>/dev/null || \
                  date -d "-${BACKUP_RETENTION_DAYS} days" +%s 2>/dev/null)

    local deleted_count=0
    while IFS= read -r -d '' backup_file; do
        local file_mtime
        file_mtime=$(stat -f %m "$backup_file" 2>/dev/null || stat -c %Y "$backup_file" 2>/dev/null)

        if [[ $file_mtime -lt $cutoff_time ]]; then
            rm -f "$backup_file"
            ((deleted_count++))
        fi
    done < <(find "$LOG_BACKUP_DIR" -name "*.tar.gz" -print0 2>/dev/null)

    if [[ $deleted_count -gt 0 ]]; then
        log_info "Deleted $deleted_count backup(s) older than $BACKUP_RETENTION_DAYS days"
    fi
}

# -----------------------------------------------------------------------------
# Process Management
# -----------------------------------------------------------------------------
pid_is_active() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
        return 1
    fi
    if ! ps -p "$pid" &>/dev/null; then
        return 1
    fi

    # Treat zombie/defunct processes as not running
    local stat
    stat=$(ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$stat" || "$stat" == *Z* ]]; then
        return 1
    fi

    return 0
}

get_app_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if pid_is_active "$pid"; then
            echo "$pid"
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    # Fallback: find by process name, prefer matching cwd
    local candidates=()
    mapfile -t candidates < <(pgrep -f "python.*app\.py" 2>/dev/null || true)

    if [[ ${#candidates[@]} -eq 0 ]]; then
        return 0
    fi

    for pid in "${candidates[@]}"; do
        local cwd
        # macOS uses lsof, Linux uses /proc
        cwd=$(lsof -p "$pid" 2>/dev/null | grep cwd | awk '{print $NF}' || \
              readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
        if [[ -n "$cwd" && "$cwd" == "$PROJECT_DIR" ]]; then
            echo "$pid"
            return 0
        fi
    done

    if [[ ${#candidates[@]} -gt 1 ]]; then
        log_warn "Multiple app-like processes found; using first PID: ${candidates[0]}"
    fi
    echo "${candidates[0]}"
    return 0
}

is_app_running() {
    local pid
    pid=$(get_app_pid)
    pid_is_active "$pid"
}

stop_app() {
    local pid
    pid=$(get_app_pid)

    if [[ -z "$pid" ]]; then
        log_info "App is not running"
        return 0
    fi

    log_info "Stopping app (PID: $pid)..."

    # Try graceful shutdown first
    kill -TERM "$pid" 2>/dev/null || true

    # Wait up to 30 seconds for graceful shutdown
    local count=0
    while ps -p "$pid" &>/dev/null && [[ $count -lt 30 ]]; do
        sleep 1
        ((count++))
    done

    # Force kill if still running
    if ps -p "$pid" &>/dev/null; then
        log_warn "App didn't stop gracefully, force killing..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 2
    fi

    rm -f "$PID_FILE"
    log_info "App stopped"
}

start_app() {
    log_info "Starting app..."

    cd "$PROJECT_DIR"
    source "$VENV_PATH/bin/activate"

    # Start app in background with separate log file
    nohup python app.py >> "$APP_LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    log_info "App started (PID: $pid), logging to $APP_LOG_FILE"

    # Wait a moment and check if process is still alive
    sleep 3
    if pid_is_active "$pid"; then
        log_info "App is running"
        return 0
    else
        log_error "App may have failed to start (process exited)"
        rm -f "$PID_FILE"
        return 1
    fi
}

restart_app() {
    log_info "Restarting app..."
    stop_app
    start_app
}

# -----------------------------------------------------------------------------
# Git Operations
# -----------------------------------------------------------------------------
has_uncommitted_changes() {
    cd "$PROJECT_DIR"
    local status
    status=$(git status --porcelain 2>/dev/null || true)
    [[ -n "$status" ]]
}

stash_changes() {
    cd "$PROJECT_DIR"
    if has_uncommitted_changes; then
        log_info "Stashing uncommitted changes..."
        git stash push -u -m "auto-deploy-$(date +%Y%m%d-%H%M%S)"
        return 0
    fi
    return 1
}

check_for_updates() {
    cd "$PROJECT_DIR"

    # Fetch latest from remote
    if ! git fetch origin 2>/dev/null; then
        log_error "git fetch failed; will retry next cycle"
        return 2
    fi

    # Get current and remote HEAD
    local local_head remote_head
    if ! local_head=$(git rev-parse HEAD 2>/dev/null); then
        log_error "Failed to resolve local HEAD"
        return 2
    fi
    if ! remote_head=$(git rev-parse '@{u}' 2>/dev/null); then
        log_warn "Could not determine remote HEAD"
        return 2
    fi

    if [[ -z "$remote_head" ]]; then
        log_warn "Could not determine remote HEAD"
        return 2
    fi

    if [[ "$local_head" != "$remote_head" ]]; then
        log_info "Updates available (local: ${local_head:0:8}, remote: ${remote_head:0:8})"
        return 0
    fi

    return 1
}

pull_changes() {
    cd "$PROJECT_DIR"
    log_info "Pulling changes..."

    local output
    if output=$(git pull 2>&1); then
        log_info "Git pull output: $output"
        return 0
    fi

    local status=$?
    log_error "Git pull failed (exit $status): $output"
    return 2
}

# -----------------------------------------------------------------------------
# Dependency Management
# -----------------------------------------------------------------------------
sync_dependencies_if_needed() {
    cd "$PROJECT_DIR"

    # Check if pyproject.toml or uv.lock changed since before the pull
    # ORIG_HEAD is set by git pull to point to where HEAD was before the pull
    local changed_files
    changed_files=$(git diff --name-only ORIG_HEAD HEAD 2>/dev/null || echo "")

    if echo "$changed_files" | grep -qE "^(pyproject\.toml|uv\.lock)$"; then
        log_info "Dependency files changed, running uv sync..."

        source "$VENV_PATH/bin/activate"
        if uv sync; then
            log_info "Dependencies synced successfully"
        else
            log_error "uv sync failed!"
            return 1
        fi
    else
        log_info "No dependency changes detected, skipping uv sync"
    fi

    return 0
}

# -----------------------------------------------------------------------------
# Main Deploy Cycle
# -----------------------------------------------------------------------------
deploy_cycle() {
    log_info "=== Starting deploy cycle ==="

    cd "$PROJECT_DIR"

    # Check for updates first (before any other operations)
    if check_for_updates; then
        log_info "Updates detected, preparing deployment..."
    else
        local status=$?
        if [[ $status -eq 2 ]]; then
            log_warn "Update check failed; retrying next cycle"
            return 0
        fi

        log_info "No updates available, skipping"
        return 0
    fi

    # Stash any local changes (note: stash is not restored after deploy)
    stash_changes || true

    # Pull changes
    if pull_changes; then
        :
    else
        local status=$?
        if [[ $status -eq 2 ]]; then
            log_warn "Git pull failed; retrying next cycle"
            return 0
        fi
        log_error "Git pull failed!"
        return 1
    fi

    # Sync dependencies if needed
    if ! sync_dependencies_if_needed; then
        log_error "Dependency sync failed!"
        # Continue anyway, app might still work
    fi

    # Restart app to pick up new code
    if is_app_running; then
        restart_app
    else
        start_app
    fi

    log_info "=== Deploy cycle complete ==="
}

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
main() {
    log_info "=============================================="
    log_info "Auto-deploy script starting"
    log_info "Project: $PROJECT_DIR"
    log_info "Poll interval: $POLL_INTERVAL seconds"
    log_info "Watchdog interval: $WATCHDOG_INTERVAL seconds"
    log_info "=============================================="

    # Rotate old logs before starting services
    rotate_logs

    # Ensure app is running on startup
    if ! is_app_running; then
        log_info "App not running, starting it..."
        start_app
    fi

    # Main loop
    local next_deploy_at
    next_deploy_at=$(date +%s)
    while true; do
        local now
        now=$(date +%s)
        if [[ $now -ge $next_deploy_at ]]; then
            if deploy_cycle; then
                :
            else
                log_error "Deploy cycle failed"
            fi
            next_deploy_at=$((now + POLL_INTERVAL))
        fi

        # Watchdog: restart app if it crashed
        if ! is_app_running; then
            log_warn "App not running (watchdog detected), restarting..."
            if ! start_app; then
                log_error "Watchdog failed to start app"
            fi
        fi

        sleep "$WATCHDOG_INTERVAL"
    done
}

# -----------------------------------------------------------------------------
# Signal Handling
# -----------------------------------------------------------------------------
cleanup() {
    log_info "Received shutdown signal, cleaning up..."
    stop_app || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Run main
main "$@"
