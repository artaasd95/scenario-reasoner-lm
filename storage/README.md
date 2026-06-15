# Storage System for Scenario Reasoner LM on RunPod

Complete S3-based data download/upload orchestration for training on RunPod with proper storage management.

## Overview

This system manages the data lifecycle for training large language models on RunPod:

- **Data Download**: Fetch datasets from S3-compatible cloud storage before training
- **Results Upload**: Periodically backup training outputs and checkpoints to cloud storage
- **Storage Management**: Handle 40GB container limit + 95GB network volume efficiently
- **Background Daemon**: Non-blocking uploads via tmux while training runs
- **State Tracking**: Avoid re-uploading unchanged files (incremental sync)

## Architecture

### Storage Layout (RunPod)

```
RunPod Pod:
  /app (40GB container root)
  ├── scenario-reasoner-lm/         (code, config)
  ├── storage/                       (this system)
  └── data → /workspace/data         (symlink to network volume)

/workspace (95GB persistent network volume)
├── data/
│   ├── raw/                         (original datasets)
│   ├── processed/                   (preprocessed data)
│   └── .download_logs/
├── experiments/
│   └── results/                     (training outputs, checkpoints)
├── artifacts/                       (demo exports)
└── runpod_suite/logs/              (daemon logs)
```

### Data Flow

```
Cloud S3 Bucket
    ↓ s3_download.sh (once, resume-capable)
Network Volume: /workspace/data
    ↓
Training Process (runs, metrics, checkpoints)
    ↓
s3_upload_daemon.sh (periodic, background)
    ↓
Cloud S3 Bucket (backup)
```

## Quick Start

### 1. Set Credentials

Create `.env` file in project root (add to `.gitignore`):

```bash
export S3_ENDPOINT_URL="https://s3.amazonaws.com"
export S3_ACCESS_KEY_ID="AKIA..."
export S3_SECRET_ACCESS_KEY="..."
export S3_BUCKET="my-training-results"
export S3_DOWNLOAD_PREFIX="datasets/scenario-v1"
export S3_UPLOAD_PREFIX="scenario-reasoner-lm"
```

### 2. Run Full Pipeline

```bash
source .env

# Download data → Start daemon → Ready for training
bash storage/orchestrate_storage.sh pipeline

# Run training (daemon uploads in background)
python runpod/train.py

# Monitor uploads
bash storage/orchestrate_storage.sh daemon:logs
```

### 3. Manual Operations

```bash
# Download only
bash storage/orchestrate_storage.sh download

# Upload only
bash storage/orchestrate_storage.sh upload

# Daemon management
bash storage/s3_upload_daemon.sh start
bash storage/s3_upload_daemon.sh status
bash storage/s3_upload_daemon.sh logs
bash storage/s3_upload_daemon.sh stop

# Status check
bash storage/orchestrate_storage.sh status
```

## Scripts

### `s3_download.sh` — Download Data

Downloads datasets from S3 to local network volume with resume support.

**Features:**
- Resume partial downloads automatically
- Keeps permanent logs in `.download_logs/`
- Logs successful downloads with metadata

**Usage:**
```bash
export S3_ENDPOINT_URL="https://s3.example.com"
export S3_ACCESS_KEY_ID="xxx"
export S3_SECRET_ACCESS_KEY="yyy"
export S3_BUCKET="training-data"
export DOWNLOAD_PREFIX="datasets/scenario-v1"
export LOCAL_DEST="./data"

bash storage/s3_download.sh
```

### `s3_upload.sh` — Upload Results

Uploads training outputs and model checkpoints to S3.

**Features:**
- Skips unchanged files (state tracking)
- Gzip compresses text files (.json, .log, .yaml, .csv) → 80-95% reduction
- Only uploads recent checkpoints (last 2)
- Retries failed uploads 3 times with 5s backoff

**Usage:**
```bash
export S3_ENDPOINT_URL="https://s3.example.com"
export S3_ACCESS_KEY_ID="xxx"
export S3_SECRET_ACCESS_KEY="yyy"
export S3_BUCKET="results-backup"
export UPLOAD_PREFIX="scenario-reasoner-lm"

bash storage/s3_upload.sh
```

### `s3_upload_daemon.sh` — Background Daemon

Runs `s3_upload.sh` periodically in tmux session (survives SSH disconnects).

**Usage:**
```bash
# Start daemon (uploads every 10 minutes)
bash storage/s3_upload_daemon.sh start

# Check status
bash storage/s3_upload_daemon.sh status

# View logs
bash storage/s3_upload_daemon.sh logs

# Attach to tmux session
bash storage/s3_upload_daemon.sh attach

# Stop daemon
bash storage/s3_upload_daemon.sh stop
```

### `orchestrate_storage.sh` — Main Orchestration

Central entry point for the complete storage pipeline.

**Usage:**
```bash
# Full pipeline
bash storage/orchestrate_storage.sh pipeline

# Individual steps
bash storage/orchestrate_storage.sh download
bash storage/orchestrate_storage.sh daemon:start
bash storage/orchestrate_storage.sh status
```

## Storage Limits & Optimization

### Container (40GB)
- Code, dependencies (avoid storing data here)
- Mount `/data` to network volume

### Network Volume (95GB)
Typical allocation for one training run:

```
├── data/           ~20GB  (datasets)
├── experiments/    ~50GB  (models + outputs)
├── artifacts/      ~10GB  (exports)
└── buffer/         ~15GB  (safety margin)
```

### Compression Benefits

| File Type | Original | Compressed | Ratio |
|-----------|----------|-----------|-------|
| metrics.json | 50MB | 2MB | 96% |
| training.log | 500MB | 100MB | 80% |

## Environment Variables

```bash
# S3 Connection (required)
S3_ENDPOINT_URL              # e.g., https://s3.amazonaws.com
S3_ACCESS_KEY_ID             # Your access key
S3_SECRET_ACCESS_KEY         # Your secret key
S3_BUCKET                    # Bucket name

# Paths (optional)
S3_DOWNLOAD_PREFIX           # Default: datasets/scenario-v1
S3_UPLOAD_PREFIX             # Default: scenario-reasoner-lm

# Daemon (optional)
UPLOAD_INTERVAL              # Default: 600 seconds (10 minutes)
```

## Files

```
storage/
├── s3_download.sh              # Download data from S3
├── s3_upload.sh                # Upload results to S3
├── s3_upload_daemon.sh         # Background daemon (tmux)
├── orchestrate_storage.sh      # Main orchestration
├── upload_state.py             # State tracking (Python)
├── .s3cmd_template             # S3cmd config template
└── README.md                   # This file
```
