#!/usr/bin/env bash
#
# s3_upload.sh — Upload training results and model checkpoints to S3
#
# Purpose: Periodically upload runs, metrics, checkpoints, and logs from the
#          RunPod network volume to S3-compatible object storage for backup
#          and long-term archival
#
# Usage:
#   export S3_ENDPOINT_URL="https://s3.example.com"
#   export S3_ACCESS_KEY_ID="xxx"
#   export S3_SECRET_ACCESS_KEY="yyy"
#   export S3_BUCKET="results-backup"
#   export UPLOAD_PREFIX="scenario-reasoner-lm"
#   export PROJECT_ROOT="."
#   bash storage/s3_upload.sh
#
# Storage Strategy:
#   - Use Python state tracker to skip unchanged files
#   - Gzip JSON/YAML/log files (80-95% reduction)
#   - Upload only new/modified content
#   - Skip large binary model files unless explicitly requested
#

set -euo pipefail

# Configuration from environment
: "${S3_ENDPOINT_URL:?Set S3_ENDPOINT_URL (e.g., https://s3.amazonaws.com)}"
: "${S3_ACCESS_KEY_ID:?Set S3_ACCESS_KEY_ID}"
: "${S3_SECRET_ACCESS_KEY:?Set S3_SECRET_ACCESS_KEY}"
: "${S3_BUCKET:?Set S3_BUCKET}"

UPLOAD_PREFIX="${UPLOAD_PREFIX:-scenario-reasoner-lm}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"
VERBOSE="${VERBOSE:-1}"
COMPRESS_TEXT="${COMPRESS_TEXT:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Paths relative to project
RUNS_DIR="${PROJECT_ROOT}/experiments/results"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts"
LOGS_DIR="${PROJECT_ROOT}/runpod_suite/logs"  # daemon logs
UPLOAD_STATE_DIR="${PROJECT_ROOT}/.upload_state"

mkdir -p "${UPLOAD_STATE_DIR}"
mkdir -p "${LOGS_DIR}"

# Setup logging
UPLOAD_LOG="${LOGS_DIR}/s3_upload_$(date +%Y%m%d_%H%M%S).log"

log() {
    local level="$1"
    shift
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${msg}" | tee -a "${UPLOAD_LOG}"
}

log "INFO" "Starting S3 upload"
log "INFO" "Endpoint: ${S3_ENDPOINT_URL}"
log "INFO" "Bucket: ${S3_BUCKET}/${UPLOAD_PREFIX}"
log "INFO" "Project root: ${PROJECT_ROOT}"
log "INFO" "Dry run: ${DRY_RUN}"

# Check prerequisites
if ! command -v s3cmd &> /dev/null; then
    log "ERROR" "s3cmd not found"
    exit 1
fi

# Create temporary s3cmd config
S3CMD_CONFIG=$(mktemp /tmp/s3cmd_ul.XXXXXX)
trap "rm -f '${S3CMD_CONFIG}'" EXIT

cat > "${S3CMD_CONFIG}" << EOF
[default]
access_key = ${S3_ACCESS_KEY_ID}
secret_key = ${S3_SECRET_ACCESS_KEY}
host_base = $(echo ${S3_ENDPOINT_URL} | sed 's|https://||g;s|http://||g')
host_bucket = %(bucket)s.$(echo ${S3_ENDPOINT_URL} | sed 's|https://||g;s|http://||g')
use_https = true
signature_version = s3v4
EOF

# Helper functions

compressible() {
    local file="$1"
    if [ "${COMPRESS_TEXT}" != "1" ]; then
        return 1
    fi
    
    case "${file##*.}" in
        json|log|yaml|yml|txt|csv|md)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

upload_file() {
    local local_path="$1"
    local remote_key="$2"
    local file_to_upload="${local_path}"
    
    if [ ! -f "${local_path}" ]; then
        return 1
    fi
    
    # Optionally compress
    if compressible "${local_path}"; then
        local compressed_tmp="${UPLOAD_STATE_DIR}/.compress_tmp_$$_$(basename ${local_path}).gz"
        if gzip -c "${local_path}" > "${compressed_tmp}" 2>/dev/null; then
            file_to_upload="${compressed_tmp}"
            remote_key="${remote_key}.gz"
        fi
    fi
    
    # Build S3 path
    local s3_path="s3://${S3_BUCKET}/${UPLOAD_PREFIX}/${remote_key}"
    
    if [ "${DRY_RUN}" = "1" ]; then
        log "DRY_RUN" "Would upload: ${local_path} -> ${s3_path}"
        return 0
    fi
    
    # Upload with retries
    local retry_count=0
    local max_retries=3
    
    while [ ${retry_count} -lt ${max_retries} ]; do
        if s3cmd \
            --config="${S3CMD_CONFIG}" \
            --no-progress \
            put "${file_to_upload}" "${s3_path}" 2>/dev/null; then
            
            log "INFO" "✓ ${remote_key}"
            
            # Cleanup compressed file
            if [ "${file_to_upload}" != "${local_path}" ]; then
                rm -f "${file_to_upload}"
            fi
            
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ ${retry_count} -lt ${max_retries} ]; then
            sleep 5
        fi
    done
    
    log "ERROR" "✗ Failed to upload: ${local_path}"
    [ "${file_to_upload}" != "${local_path}" ] && rm -f "${file_to_upload}"
    return 1
}

# Upload runs (experiments/results/runpod/...)
upload_runs() {
    if [ ! -d "${RUNS_DIR}" ]; then
        log "WARN" "Runs directory not found: ${RUNS_DIR}"
        return 0
    fi
    
    log "INFO" "Uploading from: ${RUNS_DIR}"
    
    local uploaded=0
    local failed=0
    
    # Iterate through run directories
    find "${RUNS_DIR}" -type f \( -name "*.json" -o -name "*.yaml" -o -name "*.log" -o -name "*.csv" \) | while read -r file; do
        rel_path="${file#${RUNS_DIR}/}"
        
        if upload_file "${file}" "runs/${rel_path}"; then
            uploaded=$((uploaded + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    log "INFO" "Runs upload complete"
}

# Upload model checkpoints (selective — only recent)
upload_checkpoints() {
    if [ ! -d "${RUNS_DIR}" ]; then
        return 0
    fi
    
    log "INFO" "Uploading recent checkpoints (last 2)..."
    
    # Find most recent checkpoint directories
    find "${RUNS_DIR}" -maxdepth 3 -name "checkpoint-*" -type d | sort | tail -2 | while read -r checkpoint_dir; do
        find "${checkpoint_dir}" -type f \( -name "*.bin" -o -name "*.safetensors" -o -name "*.json" \) | while read -r file; do
            rel_path="${file#${RUNS_DIR}/}"
            upload_file "${file}" "checkpoints/${rel_path}"
        done
    done
}

# Upload artifacts
upload_artifacts() {
    if [ ! -d "${ARTIFACTS_DIR}" ]; then
        return 0
    fi
    
    log "INFO" "Uploading artifacts..."
    
    find "${ARTIFACTS_DIR}" -type f | while read -r file; do
        rel_path="${file#${ARTIFACTS_DIR}/}"
        upload_file "${file}" "artifacts/${rel_path}"
    done
}

# Main upload loop
upload_start_time=$(date +%s)

# Create bucket prefix if needed
s3cmd --config="${S3CMD_CONFIG}" mb "s3://${S3_BUCKET}" 2>/dev/null || true

upload_runs
upload_artifacts
upload_checkpoints

upload_end_time=$(date +%s)
duration=$((upload_end_time - upload_start_time))

log "INFO" "Upload completed in ${duration}s"
log "INFO" "State saved to: ${UPLOAD_STATE_DIR}"

exit 0
