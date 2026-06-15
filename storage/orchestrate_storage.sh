#!/usr/bin/env bash
#
# orchestrate_storage.sh — Complete data download/upload orchestration for RunPod
#
# Purpose: Main entry point for managing data lifecycle:
#          1. Download data from cloud if needed
#          2. Start training with upload daemon in background
#          3. Monitor storage usage
#
# Usage:
#   bash storage/orchestrate_storage.sh download [data-source]
#   bash storage/orchestrate_storage.sh train [training-script]
#   bash storage/orchestrate_storage.sh upload
#   bash storage/orchestrate_storage.sh status
#   bash storage/orchestrate_storage.sh cleanup
#
# Storage Layout (RunPod):
#   /app (40GB container root)
#     └─ scenario-reasoner-lm/ (code, config)
#   /workspace (95GB network volume — mounted to /app/data in compose)
#     ├── data/
#     │   ├── raw/       (downloaded input data)
#     │   ├── processed/ (preprocessed data)
#     │   └── .download_logs/
#     ├── experiments/
#     │   └── results/   (training outputs)
#     ├── artifacts/     (demo, exports)
#     └── runpod_suite/
#         └── logs/      (daemon logs)
#

set -euo pipefail

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-.}"
WORKSPACE_MOUNT="${WORKSPACE_MOUNT:-${PROJECT_ROOT}/data}"  # Network volume mount
DATA_DIR="${WORKSPACE_MOUNT}/data"
RUNS_DIR="${WORKSPACE_MOUNT}/experiments/results"

# S3 Configuration
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"
S3_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID:-}"
S3_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY:-}"
S3_BUCKET="${S3_BUCKET:-}"
S3_DOWNLOAD_PREFIX="${S3_DOWNLOAD_PREFIX:-datasets/scenario-v1}"
S3_UPLOAD_PREFIX="${S3_UPLOAD_PREFIX:-scenario-reasoner-lm}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${WORKSPACE_MOUNT}/runpod_suite/logs"
MONITOR_LOG="${LOG_DIR}/orchestrator.log"

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${RUNS_DIR}"

# Logging
log() {
    local level="$1"
    shift
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${msg}" | tee -a "${MONITOR_LOG}"
}

# Storage monitoring
monitor_storage() {
    local path="$1"
    if [ ! -d "${path}" ]; then
        echo "0"
        return
    fi
    du -sh "${path}" 2>/dev/null | awk '{print $1}'
}

check_storage_capacity() {
    local data_size=$(monitor_storage "${DATA_DIR}")
    local runs_size=$(monitor_storage "${RUNS_DIR}")
    
    # Get available space on workspace mount
    local available=$(df "${WORKSPACE_MOUNT}" 2>/dev/null | tail -1 | awk '{print $4}')
    local available_gb=$((available / 1024 / 1024))
    
    log "INFO" "Storage status:"
    log "INFO" "  Data dir: ${data_size}"
    log "INFO" "  Runs dir: ${runs_size}"
    log "INFO" "  Available: ${available_gb}GB"
    
    if [ ${available_gb} -lt 5 ]; then
        log "WARN" "Low available space on workspace: ${available_gb}GB"
        return 1
    fi
    return 0
}

# Commands

cmd_download() {
    local data_source="${1:-cloud}"
    
    if [ -z "${S3_ENDPOINT_URL}" ]; then
        log "ERROR" "S3_ENDPOINT_URL not set"
        return 1
    fi
    
    log "INFO" "Starting data download from S3"
    log "INFO" "Source: ${S3_BUCKET}/${S3_DOWNLOAD_PREFIX}"
    log "INFO" "Destination: ${DATA_DIR}"
    
    # Export for s3_download.sh
    export S3_ENDPOINT_URL
    export S3_ACCESS_KEY_ID
    export S3_SECRET_ACCESS_KEY
    export S3_BUCKET
    export DOWNLOAD_PREFIX="${S3_DOWNLOAD_PREFIX}"
    export LOCAL_DEST="${DATA_DIR}"
    
    if bash "${SCRIPT_DIR}/s3_download.sh"; then
        log "INFO" "✓ Download completed"
        check_storage_capacity
        return 0
    else
        log "ERROR" "✗ Download failed"
        return 1
    fi
}

cmd_train() {
    local training_script="${1:-runpod/train.py}"
    
    if [ ! -f "${training_script}" ]; then
        log "ERROR" "Training script not found: ${training_script}"
        return 1
    fi
    
    log "INFO" "Starting training with background upload daemon"
    
    # Ensure upload daemon is running
    if ! bash "${SCRIPT_DIR}/s3_upload_daemon.sh" status &>/dev/null; then
        log "INFO" "Starting upload daemon..."
        if ! bash "${SCRIPT_DIR}/s3_upload_daemon.sh" start; then
            log "WARN" "Failed to start upload daemon (continuing anyway)"
        fi
    fi
    
    # Run training
    log "INFO" "Running: python ${training_script}"
    
    cd "${PROJECT_ROOT}"
    python "${training_script}"
    
    log "INFO" "✓ Training completed"
    return 0
}

cmd_upload() {
    log "INFO" "Starting manual upload pass"
    
    if [ -z "${S3_ENDPOINT_URL}" ]; then
        log "ERROR" "S3_ENDPOINT_URL not set"
        return 1
    fi
    
    # Export for s3_upload.sh
    export S3_ENDPOINT_URL
    export S3_ACCESS_KEY_ID
    export S3_SECRET_ACCESS_KEY
    export S3_BUCKET
    export UPLOAD_PREFIX="${S3_UPLOAD_PREFIX}"
    export PROJECT_ROOT
    
    if bash "${SCRIPT_DIR}/s3_upload.sh"; then
        log "INFO" "✓ Upload completed"
        return 0
    else
        log "ERROR" "✗ Upload failed"
        return 1
    fi
}

cmd_daemon_status() {
    bash "${SCRIPT_DIR}/s3_upload_daemon.sh" status
}

cmd_daemon_logs() {
    bash "${SCRIPT_DIR}/s3_upload_daemon.sh" logs
}

cmd_daemon_start() {
    bash "${SCRIPT_DIR}/s3_upload_daemon.sh" start
}

cmd_daemon_stop() {
    bash "${SCRIPT_DIR}/s3_upload_daemon.sh" stop
}

cmd_status() {
    log "INFO" "=== Orchestrator Status ==="
    log "INFO" "Project root: ${PROJECT_ROOT}"
    log "INFO" "Workspace mount: ${WORKSPACE_MOUNT}"
    
    check_storage_capacity
    
    log "INFO" ""
    log "INFO" "=== Daemon Status ==="
    cmd_daemon_status || true
    
    log "INFO" ""
    log "INFO" "=== Configuration ==="
    log "INFO" "S3 Endpoint: ${S3_ENDPOINT_URL:-<not set>}"
    log "INFO" "S3 Bucket: ${S3_BUCKET:-<not set>}"
    log "INFO" "Download prefix: ${S3_DOWNLOAD_PREFIX}"
    log "INFO" "Upload prefix: ${S3_UPLOAD_PREFIX}"
}

cmd_cleanup() {
    log "INFO" "Cleanup options:"
    log "INFO" "  1. Stop daemon: bash storage/s3_upload_daemon.sh stop"
    log "INFO" "  2. Clear local data: rm -rf ${DATA_DIR}/*"
    log "INFO" "  3. Clear runs: rm -rf ${RUNS_DIR}/*"
}

cmd_full_pipeline() {
    log "INFO" "Starting full pipeline"
    
    # Check prerequisites
    if [ -z "${S3_ENDPOINT_URL}" ]; then
        log "ERROR" "S3_ENDPOINT_URL not set. Set environment variables:"
        log "ERROR" "  export S3_ENDPOINT_URL='https://...'"
        log "ERROR" "  export S3_ACCESS_KEY_ID='...'"
        log "ERROR" "  export S3_SECRET_ACCESS_KEY='...'"
        log "ERROR" "  export S3_BUCKET='...'"
        return 1
    fi
    
    # Step 1: Download data
    log "INFO" ""
    log "INFO" "STEP 1: Download data"
    if cmd_download; then
        log "INFO" "✓ Download successful"
    else
        log "ERROR" "✗ Download failed"
        return 1
    fi
    
    # Step 2: Start daemon
    log "INFO" ""
    log "INFO" "STEP 2: Start upload daemon"
    if cmd_daemon_start; then
        log "INFO" "✓ Daemon started"
    else
        log "ERROR" "✗ Failed to start daemon"
        return 1
    fi
    
    # Step 3: Show status
    log "INFO" ""
    log "INFO" "STEP 3: Ready for training"
    cmd_status
    
    log "INFO" ""
    log "INFO" "Pipeline ready. Run training:"
    log "INFO" "  python runpod/train.py"
    log "INFO" ""
    log "INFO" "Monitor uploads:"
    log "INFO" "  bash storage/s3_upload_daemon.sh logs"
}

# Main dispatcher
COMMAND="${1:-status}"
shift || true

case "${COMMAND}" in
    download)
        cmd_download "$@"
        ;;
    train)
        cmd_train "$@"
        ;;
    upload)
        cmd_upload
        ;;
    daemon:status)
        cmd_daemon_status
        ;;
    daemon:logs)
        cmd_daemon_logs
        ;;
    daemon:start)
        cmd_daemon_start
        ;;
    daemon:stop)
        cmd_daemon_stop
        ;;
    status)
        cmd_status
        ;;
    cleanup)
        cmd_cleanup
        ;;
    pipeline)
        cmd_full_pipeline
        ;;
    *)
        cat << USAGE
Usage: bash storage/orchestrate_storage.sh [COMMAND] [OPTIONS]

Commands:
    download          Download data from S3 to local workspace
    train [SCRIPT]    Run training with background upload daemon
    upload            Perform manual upload pass
    daemon:start      Start upload daemon
    daemon:stop       Stop upload daemon
    daemon:status     Check daemon status
    daemon:logs       Show daemon logs
    status            Show overall status
    cleanup           Show cleanup instructions
    pipeline          Run full pipeline (download → daemon → ready)

Environment Variables:
    S3_ENDPOINT_URL       S3 endpoint (e.g., https://s3.amazonaws.com)
    S3_ACCESS_KEY_ID      S3 access key
    S3_SECRET_ACCESS_KEY  S3 secret key
    S3_BUCKET             S3 bucket name
    S3_DOWNLOAD_PREFIX    Path in bucket for downloads (default: datasets/scenario-v1)
    S3_UPLOAD_PREFIX      Path in bucket for uploads (default: scenario-reasoner-lm)
    PROJECT_ROOT          Project root directory (default: .)
    WORKSPACE_MOUNT       Network volume mount (default: ./data)

Examples:
    # Full pipeline
    export S3_ENDPOINT_URL="https://s3.example.com"
    export S3_ACCESS_KEY_ID="xxx"
    export S3_SECRET_ACCESS_KEY="yyy"
    export S3_BUCKET="training-data"
    bash storage/orchestrate_storage.sh pipeline
    
    # Manual steps
    bash storage/orchestrate_storage.sh download
    bash storage/orchestrate_storage.sh daemon:start
    python runpod/train.py
    bash storage/orchestrate_storage.sh status
    bash storage/orchestrate_storage.sh daemon:logs

USAGE
        exit 1
        ;;
esac
