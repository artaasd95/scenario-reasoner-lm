# Scenario Reasoner LM

A training suite for open-source reasoning LLMs to reason into scenarios. This project provides the foundational infrastructure — base metrics, losses, datasets, monitoring (CoT/ToT/Aha), and logging — ready for further development of scenario-based reasoning capabilities.

## Overview

`scenario-reasoner-lm` is built to train language models to reason deeply into structured scenarios. It provides:

- **Base Metrics** — Extensible metric classes for evaluating reasoning quality
- **Base Losses** — Modular loss function base classes for custom training objectives
- **Data Infrastructure** — Supports both HuggingFace and custom datasets/models with PyTorch
- **Monitoring** — Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Aha Moment catchers for tracking reasoning signals during training
- **Logging** — Local file logging and Weights & Biases integration

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the complete folder layout.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Components

### Data
- `src/data/base_dataset.py` — Abstract base dataset
- `src/data/hf_datasets.py` — HuggingFace dataset/model wrappers
- `src/data/custom_datasets.py` — Custom PyTorch dataset classes
- `src/data/evaluators.py` — Dataset evaluation utilities

### Losses
- `src/losses/base_losses.py` — Base loss class hierarchy

### Metrics
- `src/metrics/base_metrics.py` — Base metric class hierarchy

### Monitoring
- `src/monitoring/cot_monitor.py` — Chain-of-Thought pattern detection
- `src/monitoring/tot_monitor.py` — Tree-of-Thought pattern detection
- `src/monitoring/aha_monitor.py` — Aha-moment detection in reasoning traces

### Logging
- `src/logging/local_logger.py` — Local file-based logging
- `src/logging/wandb_logger.py` — Weights & Biases logging

## License

Apache License 2.0 — Copyright 2026 artaasd95
A training suite for LLM to reasoning in the different scenarios
