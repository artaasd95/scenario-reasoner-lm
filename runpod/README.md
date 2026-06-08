# RunPod execution — scenario-reasoner-lm

Tmux-safe Unsloth DPO training with auto-resume and FTP storage sync.

## Quick start

```bash
tmux new -s train
bash runpod/setup.sh
python runpod/train.py
```

## Resume

```bash
bash runpod/resume.sh
```

Wraps `scripts/train.py --config runpod/config.yaml`.

See `storage/README.md` for FTP environment variables.
