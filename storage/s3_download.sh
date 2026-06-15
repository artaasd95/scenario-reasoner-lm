#!/usr/bin/env bash
#
# s3_download.sh — Download data from S3-compatible bucket to local network volume
#
# Purpose: Fetch datasets, models, or other dependencies from cloud object storage
#          to the RunPod network volume before training
#
# Usage:
#   export S3_ENDPOINT_URL="https://s3.example.com"
#   export S3_ACCESS_KEY_ID="xxx"
#   export S3_SECRET_ACCESS_KEY="yyy"
#   export S3_BUCKET="training-data"
#   export DOWNLOAD_PREFIX="datasets/scenario-v1"
#   export LOCAL_DEST="./data"
#   bash storage/s3_download.sh
#
# Storage Strategy:
#   - Download to ./data (mounted on 95GB network volume)
#   - Never download to container root (/app — 40GB limit)
#   - Resume partial downloads automatically
#

set -euo pipefail

# Configuration from environment
: "${S3_ENDPOINT_URL:?Set S3_ENDPOINT_URL (e.g., https://s3.amazonaws.com)}"
: "${S3_ACCESS_KEY_ID:?Set S3_ACCESS_KEY_ID}"
: "${S3_SECRET_ACCESS_KEY:?Set S3_SECRET_ACCESS_KEY}"
: "${S3_BUCKET:?Set S3_BUCKET}"

DOWNLOAD_PREFIX="${DOWNLOAD_PREFIX:-}"
LOCAL_DEST="${LOCAL_DEST:-./data}"
VERBOSE="${VERBOSE:-1}"

# Setup logging
LOG_DIR="${LOCAL_DEST}/.download_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/download_$(date +%Y%m%d_%H%M%S).log"

log() {
    local level="$1"
    shift
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${msg}" | tee -a "${LOG_FILE}"
}

log "INFO" "Starting S3 download"
log "INFO" "Endpoint: ${S3_ENDPOINT_URL}"
log "INFO" "Bucket: ${S3_BUCKET}"
log "INFO" "Prefix: ${DOWNLOAD_PREFIX:-<root>}"
log "INFO" "Destination: ${LOCAL_DEST}"

# Check prerequisites
if ! command -v s3cmd &> /dev/null; then
    log "ERROR" "s3cmd not found. Install it: pip install s3cmd or apt-get install s3cmd"
    exit 1
fi

mkdir -p "${LOCAL_DEST}"

# Create temporary s3cmd config file (never commit credentials to repo)
S3CMD_CONFIG=$(mktemp /tmp/s3cmd_dl.XXXXXX)
trap "rm -f '${S3CMD_CONFIG}'" EXIT

# Build s3cmd config (minimal)
cat > "${S3CMD_CONFIG}" << EOF
[default]
access_key = ${S3_ACCESS_KEY_ID}
secret_key = ${S3_SECRET_ACCESS_KEY}
host_base = $(echo ${S3_ENDPOINT_URL} | sed 's|https://||g;s|http://||g')
host_bucket = %(bucket)s.$(echo ${S3_ENDPOINT_URL} | sed 's|https://||g;s|http://||g')
use_https = true
signature_version = s3v4
EOF

log "INFO" "Configuration ready"

# Build S3 path
S3_PATH="s3://${S3_BUCKET}"
if [ -n "${DOWNLOAD_PREFIX}" ]; then
    S3_PATH="${S3_PATH}/${DOWNLOAD_PREFIX}"
fi

log "INFO" "Source S3 path: ${S3_PATH}"

# Download with resume support
# Flags explained:
#   --delete-removed: Remove local files not in bucket
#   --skip-existing: Skip files already present locally (resume)
#   --multipart-chunk-size=10485760: 10MB chunks for parallel transfer
#   -r: Recursive
#   -v: Verbose
#   -m: Show progress (multipart if available)

if [ "${VERBOSE}" = "1" ]; then
    VERBOSE_FLAG="-v"
else
    VERBOSE_FLAG=""
fi

download_start_time=$(date +%s)

if s3cmd \
    --config="${S3CMD_CONFIG}" \
    --multipart-chunk-size=10485760 \
    --skip-existing \
    ${VERBOSE_FLAG} \
    sync \
    "${S3_PATH}" \
    "${LOCAL_DEST}"; then
    
    download_end_time=$(date +%s)
    duration=$((download_end_time - download_start_time))
    
    # Get local directory size
    local_size=$(du -sh "${LOCAL_DEST}" 2>/dev/null | awk '{print $1}')
    
    log "INFO" "Download completed successfully in ${duration}s"
    log "INFO" "Local data size: ${local_size}"
    
    # Save download metadata
    cat > "${LOG_DIR}/download_metadata.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "s3_bucket": "${S3_BUCKET}",
  "s3_prefix": "${DOWNLOAD_PREFIX}",
  "s3_endpoint": "${S3_ENDPOINT_URL}",
  "local_dest": "${LOCAL_DEST}",
  "local_size": "${local_size}",
  "duration_seconds": ${duration},
  "success": true
}
EOF
    
    exit 0
else
    download_end_time=$(date +%s)
    duration=$((download_end_time - download_start_time))
    
    log "ERROR" "Download failed after ${duration}s"
    log "ERROR" "Check S3 credentials and network connectivity"
    
    # Save failure metadata
    cat > "${LOG_DIR}/download_metadata.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "s3_bucket": "${S3_BUCKET}",
  "s3_prefix": "${DOWNLOAD_PREFIX}",
  "s3_endpoint": "${S3_ENDPOINT_URL}",
  "local_dest": "${LOCAL_DEST}",
  "duration_seconds": ${duration},
  "success": false
}
EOF
    
    exit 1
fi
