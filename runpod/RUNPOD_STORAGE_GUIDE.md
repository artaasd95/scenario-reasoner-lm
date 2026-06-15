# RunPod Storage & Training Guide

Complete guide for running Scenario Reasoner LM training on RunPod with S3-based data management.

## Overview

This guide covers:
- Setting up S3 credentials for data download/upload
- Downloading training data from cloud storage
- Running training with background result uploads
- Monitoring storage usage and upload progress
- Managing the RunPod pod efficiently

## RunPod Pod Setup

### Step 1: Launch Pod

**Recommended Configuration:**
- Image: CUDA 12.1 (or latest PyTorch image)
- GPU: A100 or H100 (40GB VRAM recommended)
- Storage: 
  - Container root: 40GB (included)
  - Network volume: 95GB (add to pod config)
- Port: 8000 (for API serving)

### Step 2: Mount Network Volume

On the RunPod pod configuration:
```yaml
volumes:
  - name: workspace
    mount_path: /workspace
    size: 95GB
```

The pod will expose this as `/workspace` inside the container.

### Step 3: Clone Repository

```bash
cd /app  # Container root
git clone https://github.com/artaasd95/scenario-reasoner-lm.git
cd scenario-reasoner-lm
```

## Storage Configuration

### Step 1: Create Credentials File

Run the interactive setup:

```bash
bash runpod/storage_setup.sh
```

Or manually create `.env`:

```bash
cat > .env << 'EOF'
export S3_ENDPOINT_URL="https://s3.amazonaws.com"
export S3_ACCESS_KEY_ID="AKIA..."
export S3_SECRET_ACCESS_KEY="..."
export S3_BUCKET="my-results-bucket"
export S3_DOWNLOAD_PREFIX="datasets/scenario-v1"
export S3_UPLOAD_PREFIX="scenario-reasoner-lm"
EOF

chmod 600 .env
```

### Step 2: Test S3 Connection

```bash
source .env

# Try downloading a small test file
export DRY_RUN=1
bash storage/s3_download.sh

# Should show: [INFO] Download completed successfully
```

### Step 3: Verify Storage Paths

```bash
# Check network volume is accessible
df -h /workspace

# Create required directories
mkdir -p /workspace/data
mkdir -p /workspace/experiments/results
mkdir -p /workspace/artifacts

# Link from /app/data to /workspace/data
ln -sfn /workspace/data /app/data
```

## Training Workflow

### Quick Start (Automated)

```bash
cd /app/scenario-reasoner-lm
source .env

# One command: download → daemon start → training ready
bash storage/orchestrate_storage.sh pipeline

# Once ready, run training
python runpod/train.py --config runpod/config.yaml

# In another terminal, monitor uploads
bash storage/s3_upload_daemon.sh logs
```

### Manual Steps

#### Step 1: Download Data

```bash
source .env
bash storage/orchestrate_storage.sh download

# Output:
# [2025-06-13 12:00:30] [INFO] Starting S3 download
# [2025-06-13 12:05:45] [INFO] Download completed successfully in 315s
# [2025-06-13 12:05:45] [INFO] Local data size: 25.4G
```

Expected files:
```
/workspace/data/
├── raw/              (original datasets)
├── processed/        (preprocessed)
└── .download_logs/   (metadata)
```

#### Step 2: Start Upload Daemon

```bash
bash storage/s3_upload_daemon.sh start

# Verify it's running
bash storage/s3_upload_daemon.sh status
# Output: [INFO] Daemon is RUNNING
```

The daemon will:
- Upload new/changed files every 10 minutes
- Compress JSON/YAML/log files (80-95% reduction)
- Skip unchanged files (state tracking)
- Survive SSH disconnections (tmux)

#### Step 3: Run Training

```bash
python runpod/train.py \
    --config runpod/config.yaml \
    --output experiments/results/runpod \
    --num-epochs 50 \
    --batch-size 4
```

During training:
- Checkpoints are saved to `experiments/results/runpod/checkpoints/`
- Metrics are saved as JSON to `experiments/results/runpod/metrics/`
- Logs are written to `experiments/results/runpod/logs/`
- Daemon automatically uploads these in background

#### Step 4: Monitor Progress

In another SSH session:

```bash
# Check storage usage
bash storage/orchestrate_storage.sh status
# Output:
# Storage status:
#   Data dir: 25.4G
#   Runs dir: 12.3G
#   Available: 57GB

# Watch upload progress
bash storage/s3_upload_daemon.sh logs

# Or attach to daemon session
bash storage/s3_upload_daemon.sh attach
# [Ctrl+B then D] to detach from tmux
```

#### Step 5: After Training

The daemon will continue uploading any remaining files:

```bash
# Check final upload status
bash storage/s3_upload_daemon.sh logs

# Optional: Stop daemon
bash storage/s3_upload_daemon.sh stop

# Verify files in S3
source .env
s3cmd --help  # Install if needed, then list bucket
```

## Docker Compose Deployment

For running entire pipeline in Docker:

```bash
# Build image
docker-compose -f docker-compose.runpod.yml build scenario-train

# Run training (handles everything automatically)
docker-compose -f docker-compose.runpod.yml run --rm \
    -e S3_ENDPOINT_URL="https://s3.amazonaws.com" \
    -e S3_ACCESS_KEY_ID="AKIA..." \
    -e S3_SECRET_ACCESS_KEY="..." \
    -e S3_BUCKET="my-bucket" \
    -v /workspace:/app/data \
    scenario-train

# Or use .env file
docker-compose -f docker-compose.runpod.yml run --rm \
    --env-file .env \
    -v /workspace:/app/data \
    scenario-train
```

## Storage Management

### Monitoring Disk Usage

```bash
# Real-time storage status
bash storage/orchestrate_storage.sh status

# Detailed breakdown
du -sh /workspace/*

# Available space
df -h /workspace

# What's using space
du -sh /workspace/experiments/results/*
du -sh /workspace/data/*
```

### Typical Storage Usage

For a complete training run (50 epochs on 4 GPUs):

```
Data download:     20GB  (one-time)
Checkpoints:       30GB  (last 2 saved)
Metrics/Logs:       5GB  (compressed)
Total:            ~55GB  (fits in 95GB volume)
```

### Cleanup

If running low on space:

```bash
# Remove old checkpoints (keep last 2)
find /workspace/experiments -name "checkpoint-*" -type d | sort | head -n -2 | xargs rm -rf

# Clear download cache (WARNING: will need to re-download)
# rm -rf /workspace/data/raw/*

# Final upload of remaining files
bash storage/s3_upload.sh

# Stop daemon
bash storage/s3_upload_daemon.sh stop
```

## Troubleshooting

### S3 Connection Fails

```bash
# Check credentials
echo $S3_ACCESS_KEY_ID
echo $S3_SECRET_ACCESS_KEY

# Test s3cmd directly
s3cmd --configure   # Interactive setup
s3cmd ls s3://your-bucket
```

### Daemon Not Uploading

```bash
# Check daemon status
bash storage/s3_upload_daemon.sh status

# View recent logs
bash storage/s3_upload_daemon.sh logs | tail -20

# Check if tmux session exists
tmux list-sessions

# Restart daemon
bash storage/s3_upload_daemon.sh stop
sleep 1
bash storage/s3_upload_daemon.sh start
```

### Out of Space

```bash
# Check what's taking space
du -sh /workspace/experiments/results/*
du -sh /workspace/data/*

# Force immediate upload
bash storage/s3_upload.sh

# Then cleanup local files
rm -rf /workspace/experiments/results/run_old_experiment/
```

### Slow Upload/Download

```bash
# Check network
ping -c 3 8.8.8.8

# Check available bandwidth
iftop  # or nethogs

# Adjust upload interval
export UPLOAD_INTERVAL=300  # Upload every 5 minutes instead of 10
bash storage/s3_upload_daemon.sh restart
```

## Performance Tips

### For Faster Training

1. **Increase GPU utilization:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all GPUs
   python runpod/train.py --batch-size 32 --gradient-accumulation 4
   ```

2. **Optimize data loading:**
   ```bash
   export NUM_WORKERS=8  # CPU workers for data loading
   python runpod/train.py --num-workers 8
   ```

3. **Use mixed precision training:**
   ```bash
   python runpod/train.py --bf16  # BFloat16 for A100/H100
   ```

### For Faster Uploads

1. **Adjust compression:**
   ```bash
   export COMPRESS_TEXT=1  # Reduce from 200MB to 10MB
   ```

2. **Increase upload frequency:**
   ```bash
   export UPLOAD_INTERVAL=300  # Upload every 5 min instead of 10
   ```

3. **Use faster S3 endpoint:**
   - Choose S3 region closest to RunPod datacenter
   - Or use local S3-compatible storage (MinIO)

## Monitoring Checklist

During training, periodically check:

- [ ] Upload daemon is running: `bash storage/s3_upload_daemon.sh status`
- [ ] Training is progressing: `nvidia-smi` (GPU utilization > 80%)
- [ ] Storage is not full: `df -h /workspace` (> 10GB free)
- [ ] Results are uploading: `bash storage/s3_upload_daemon.sh logs`
- [ ] No errors in training: `tail -f /workspace/experiments/results/runpod/logs/*.log`

## Complete Example Session

```bash
# --- Terminal 1: Setup and Training ---
cd /app/scenario-reasoner-lm
source .env

# Download data (15-30 minutes)
bash storage/orchestrate_storage.sh download

# Start daemon (background, continues forever)
bash storage/s3_upload_daemon.sh start

# Run training (30-120 minutes depending on config)
python runpod/train.py --config runpod/config.yaml --num-epochs 50

# --- Terminal 2: Monitoring (while training runs) ---
cd /app/scenario-reasoner-lm

# Watch uploads
watch -n 10 'bash storage/s3_upload_daemon.sh logs | tail -20'

# Or check storage every 5 minutes
while true; do
    clear
    echo "=== Storage Status ==="
    df -h /workspace | tail -1
    echo ""
    echo "=== Directory Sizes ==="
    du -sh /workspace/*
    echo ""
    echo "=== Last Uploads ==="
    bash storage/s3_upload_daemon.sh logs | tail -5
    sleep 300
done

# --- After Training ---

# Verify all files uploaded
bash storage/s3_upload_daemon.sh logs

# Stop daemon
bash storage/s3_upload_daemon.sh stop

# Download results from S3 to local machine
aws s3 sync s3://my-bucket/scenario-reasoner-lm/ ./results/
```

## Next Steps

1. **Evaluate results locally:**
   ```bash
   python scripts/evaluate.py --runs-dir results/
   ```

2. **Create demo:**
   ```bash
   python scripts/export_demo_artifacts.py --run-dir results/
   ```

3. **Deploy API:**
   ```bash
   docker-compose -f docker-compose.yml up scenario-api
   ```

## Support

- **Storage documentation:** `storage/README.md`
- **Configuration template:** `storage/.env.example`
- **RunPod docs:** https://docs.runpod.io/
- **S3 docs:** https://aws.amazon.com/s3/

For issues:
1. Check logs: `bash storage/s3_upload_daemon.sh logs`
2. Test S3: `bash storage/orchestrate_storage.sh status`
3. Review credentials: `source .env && echo $S3_BUCKET`
