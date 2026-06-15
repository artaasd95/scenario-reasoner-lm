#!/usr/bin/env bash
#
# s3_upload_daemon.sh — Continuous background upload daemon using tmux
#
# Purpose: Run s3_upload.sh periodically in background without blocking training.
#          Uses tmux for process management and survival across SSH disconnections.
#
# Usage:
#   export S3_ENDPOINT_URL="https://s3.example.com"
#   export S3_ACCESS_KEY_ID="xxx"
#   export S3_SECRET_ACCESS_KEY="yyy"
#   export S3_BUCKET="results-backup"
#   bash storage/s3_upload_daemon.sh start
#
# Commands:
#   bash storage/s3_upload_daemon.sh start    - Start daemon
#   bash storage/s3_upload_daemon.sh stop     - Stop daemon
#   bash storage/s3_upload_daemon.sh status   - Check status
#   bash storage/s3_upload_daemon.sh logs     - Tail logs
#

set -euo pipefail

COMMAND="${1:-status}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"
UPLOAD_INTERVAL="${UPLOAD_INTERVAL:-600}"  # 10 minutes default

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOGS_DIR="${PROJECT_ROOT}/runpod_suite/logs"
mkdir -p "${LOGS_DIR}"

SESSION_NAME="s3-upload-daemon"
LOG_FILE="${LOGS_DIR}/upload_daemon.log"

log() {
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${msg}" | tee -a "${LOG_FILE}"
}

start_daemon() {
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log "ERROR: Daemon already running (session: ${SESSION_NAME})"
        return 1
    fi
    
    log "INFO: Starting upload daemon (interval: ${UPLOAD_INTERVAL}s)"
    
    # Build daemon command
    # This runs s3_upload.sh periodically in a tmux session
    DAEMON_CMD=$(cat << 'EOF'
#!/usr/bin/env bash
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

log_dir="${PROJECT_ROOT}/runpod_suite/logs"
mkdir -p "${log_dir}"

log_file="${log_dir}/upload_daemon.log"
log() {
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${msg}" | tee -a "${log_file}"
}

log "Daemon started (PID: $$)"

while true; do
    log "Running upload pass..."
    bash "${SCRIPT_DIR}/s3_upload.sh" 2>&1 | tee -a "${log_file}"
    
    sleep_time=${UPLOAD_INTERVAL:-600}
    log "Next upload in ${sleep_time}s..."
    sleep ${sleep_time}
done
EOF
)
    
    # Create temporary script
    DAEMON_SCRIPT=$(mktemp /tmp/s3_daemon.XXXXXX)
    echo "${DAEMON_CMD}" > "${DAEMON_SCRIPT}"
    chmod +x "${DAEMON_SCRIPT}"
    
    # Start tmux session with daemon
    tmux new-session -d -s "${SESSION_NAME}" \
        "cd '${PROJECT_ROOT}' && bash '${DAEMON_SCRIPT}'" \
        2>&1 | tee -a "${LOG_FILE}"
    
    sleep 1
    
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log "INFO: Daemon started successfully"
        log "INFO: Session: ${SESSION_NAME}"
        log "INFO: Logs: ${LOG_FILE}"
        log "INFO: View: tmux attach-session -t ${SESSION_NAME}"
        log "INFO: Stop: bash storage/s3_upload_daemon.sh stop"
        return 0
    else
        log "ERROR: Failed to start daemon"
        return 1
    fi
}

stop_daemon() {
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log "WARN: Daemon not running (no session: ${SESSION_NAME})"
        return 0
    fi
    
    log "INFO: Stopping daemon..."
    tmux kill-session -t "${SESSION_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    log "INFO: Daemon stopped"
    return 0
}

check_status() {
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log "INFO: Daemon is RUNNING"
        log "INFO: Session: ${SESSION_NAME}"
        tmux list-windows -t "${SESSION_NAME}"
        return 0
    else
        log "WARN: Daemon is NOT running"
        return 1
    fi
}

show_logs() {
    if [ ! -f "${LOG_FILE}" ]; then
        log "WARN: No log file yet: ${LOG_FILE}"
        return 1
    fi
    
    # Show last 50 lines
    tail -50 "${LOG_FILE}"
}

attach() {
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log "ERROR: Daemon not running"
        return 1
    fi
    
    tmux attach-session -t "${SESSION_NAME}"
}

# Main
case "${COMMAND}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    attach)
        attach
        ;;
    restart)
        stop_daemon
        sleep 1
        start_daemon
        ;;
    *)
        cat << USAGE
Usage: bash storage/s3_upload_daemon.sh [COMMAND]

Commands:
    start      Start the upload daemon
    stop       Stop the daemon
    status     Check daemon status
    logs       Show daemon logs (last 50 lines)
    attach     Attach to tmux session
    restart    Restart the daemon

Environment:
    UPLOAD_INTERVAL    Seconds between uploads (default: 600)
    PROJECT_ROOT       Project root directory (default: current dir)

Examples:
    bash storage/s3_upload_daemon.sh start
    UPLOAD_INTERVAL=300 bash storage/s3_upload_daemon.sh start
    bash storage/s3_upload_daemon.sh logs
    bash storage/s3_upload_daemon.sh attach

USAGE
        exit 1
        ;;
esac
