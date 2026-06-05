# Operations Runbook

## Single-process default (CI)

Training and eval default to **single GPU/CPU process**. Distributed training is opt-in:

```python
from src.training.distributed.config import DistributedTrainingConfig
cfg = DistributedTrainingConfig.single_process()
```

Enable DDP/FSDP only in research jobs; CI must call `validate_ci_safe()` or keep `enabled=False`.

## Data platform builds

See [data-platform.md](data-platform.md).

## Experiment logging

```python
from src.logging.experiment_logger import ExperimentLoggerFactory
logger = ExperimentLoggerFactory.create("local", name="run", log_dir="./logs")
logger.log_metrics({"loss": 0.5}, step=1)
logger.close()
```

Backends: `local` (default), `wandb`, `comet` (stub JSONL), `sql` (JSONL shipper).

## Feedback loop (stub → S10-07)

```bash
python scripts/apply_feedback.py --config configs/data/feedback.yaml
```

Place JSONL rows in `feedback/incoming/` with `record_id`, `coherence_score`, and optional `theta`.
