# Scenario Reasoner LM — Project Structure

This document describes the complete folder structure of the Scenario Reasoner LM project.

## Overview

`scenario-reasoner-lm` is a training-suite scaffold for open-source reasoning LLMs to reason into scenarios. The project provides bases and infrastructure ready for further development.

## Complete Structure

```
scenario-reasoner-lm/
├── README.md                          # Project overview
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # Apache 2.0, 2026
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── data/                          # Data loading and evaluation
│   │   ├── __init__.py
│   │   ├── base_dataset.py           # Abstract base dataset class
│   │   ├── hf_datasets.py            # HuggingFace dataset/model wrappers
│   │   ├── custom_datasets.py        # Custom PyTorch dataset classes
│   │   └── evaluators.py             # Dataset evaluation utilities
│   │
│   ├── losses/                        # Loss functions
│   │   ├── __init__.py
│   │   └── base_losses.py            # Base loss class hierarchy
│   │
│   ├── metrics/                       # Evaluation metrics
│   │   ├── __init__.py
│   │   └── base_metrics.py           # Base metric class hierarchy
│   │
│   ├── monitoring/                    # Reasoning monitoring
│   │   ├── __init__.py
│   │   ├── cot_monitor.py            # Chain-of-Thought pattern detector
│   │   ├── tot_monitor.py            # Tree-of-Thought pattern detector
│   │   └── aha_monitor.py            # Aha-moment catcher
│   │
│   └── logging/                       # Logging suite
│       ├── __init__.py
│       ├── local_logger.py           # Local file-based logger
│       └── wandb_logger.py           # Weights & Biases logger
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Data loading and exploration
│   ├── 02_monitoring_demo.ipynb      # CoT/ToT/Aha monitoring demo
│   └── 03_metrics_and_losses.ipynb   # Metrics and losses demo
│
├── experiments/                       # Experiment configs and results
│   ├── configs/
│   │   └── example_config.json       # Example configuration
│   └── results/
│       └── .gitkeep
│
├── data/                              # Data storage
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── tests/                             # Testing suite
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_datasets.py          # Dataset unit tests
│   │   ├── test_losses.py            # Loss unit tests
│   │   ├── test_metrics.py           # Metric unit tests
│   │   └── test_monitoring.py        # Monitoring unit tests
│   └── integration/
│       ├── __init__.py
│       └── test_pipeline.py          # Integration pipeline tests
│
├── scripts/                           # Helper scripts
│   ├── train.py                      # Training entry point (scaffold)
│   └── evaluate.py                   # Evaluation entry point (scaffold)
│
└── docs/                              # Documentation
    └── ideas-plan.md                 # Project plan and ideas
```

## Key Components

### Data (`src/data/`)
| File | Purpose |
|------|---------|
| `base_dataset.py` | Abstract base class all datasets must inherit from |
| `hf_datasets.py` | Wraps HuggingFace `datasets` and `transformers` objects into torch-compatible form |
| `custom_datasets.py` | Custom scenario-based torch Dataset implementations |
| `evaluators.py` | Tools for evaluating dataset quality (coverage, diversity, etc.) |

### Losses (`src/losses/`)
| File | Purpose |
|------|---------|
| `base_losses.py` | Abstract `BaseLoss(nn.Module, ABC)` + token-level and sequence-level base shells |

### Metrics (`src/metrics/`)
| File | Purpose |
|------|---------|
| `base_metrics.py` | Abstract `BaseMetric` + registry pattern for metric management |

### Monitoring (`src/monitoring/`)
| File | Purpose |
|------|---------|
| `cot_monitor.py` | Detects and logs Chain-of-Thought patterns in LLM outputs |
| `tot_monitor.py` | Detects and logs Tree-of-Thought branching in LLM outputs |
| `aha_monitor.py` | Catches "aha moment" signals (sudden insight patterns) in reasoning traces |

### Logging (`src/logging/`)
| File | Purpose |
|------|---------|
| `local_logger.py` | Structured local file and console logging |
| `wandb_logger.py` | Weights & Biases experiment tracker integration |

## Status

### ✅ Ready (Scaffold)
- Full folder structure
- All base/abstract classes with documented interfaces
- HuggingFace + custom dataset wrappers
- CoT / ToT / Aha monitoring infrastructure
- Local + W&B logging suite
- Unit & integration test stubs
- Example notebooks
- Example experiment config

### 🔨 To Be Implemented
All concrete model architectures, actual training methods, and domain-specific
scenario logic are intentionally left for future development iterations.
