# GNN Quant Qlib

Qlib + PyTorch Temporal GCN research project for cross-sectional equity ranking.

This repository implements a Qlib-compatible graph neural network model that treats each trading day as a stock graph. It combines a temporal encoder for Alpha360 features with graph convolution over dynamically constructed stock relationships, then plugs into Qlib's standard workflow for training, signal analysis, and portfolio backtesting.

> Research code only. This project is not investment advice.

## Highlights

- Qlib-native model wrapper with `fit`, `predict`, `save`, and `load` support.
- Temporal encoder based on GRU or LSTM for Alpha360 rolling features.
- Daily cross-sectional graph construction from embedding cosine similarity.
- Sparse top-k graph convolution with residual prediction head.
- GPU training support through the `GPU` model argument.
- Optuna tuning script for reproducible hyperparameter search.
- Lightweight unit tests that stub Qlib where possible, so CI can validate model logic without downloading market data.

## Architecture

The model follows this flow:

```text
Alpha360 features
    -> GRU/LSTM temporal encoder
    -> dynamic stock graph from daily embeddings
    -> two-layer GCN
    -> residual MLP head
    -> cross-sectional prediction score
```

Each daily batch is a graph:

- node: one stock in the daily universe
- node feature: Alpha360 historical feature vector
- edge: non-negative cosine similarity between encoded stock embeddings
- target: next-period return label from the Qlib workflow

The Qlib workflow then records predictions, signal IC, and TopK-Dropout portfolio analysis.

## Repository Layout

```text
qlib_gcn_model/
  qlib_gcn_model.py       # TemporalGCNNet and QlibGCN wrapper
examples/
  workflow_config_gcn_Alpha360.yaml
scripts/
  tune_optuna.py          # Optuna tuning entrypoint
docs/
  optuna_tuning.md        # Tuning usage notes
tests/
  test_qlib_gcn_model.py  # Lightweight model tests
```

## Installation

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For GPU training, install a PyTorch build that matches your local CUDA driver. The project itself selects CUDA through the workflow config:

```yaml
GPU: 0
```

Use `GPU: -1` or `GPU: cpu` to force CPU mode.

## Data

The example workflow expects Qlib CN daily data at:

```text
~/.qlib/qlib_data/cn_data
```

The dataset is not included in this repository. Prepare Qlib data separately, then run the workflow from the repository root.

## Training

```bash
qrun examples/workflow_config_gcn_Alpha360.yaml
```

The default example uses:

- market: CSI300
- benchmark: SH000300
- train: 2008-01-01 to 2014-12-31
- valid: 2015-01-01 to 2016-12-31
- test/backtest: 2017-01-01 to 2020-08-01
- model: GRU + dynamic Temporal GCN
- early stopping metric: daily cross-sectional IC

## Optuna Tuning

Run a short GPU tuning study:

```bash
python scripts/tune_optuna.py --gpu 0 --n-trials 20 --trial-epochs 50 --trial-early-stop 10
```

Save the best merged workflow config:

```bash
python scripts/tune_optuna.py \
  --gpu 0 \
  --n-trials 20 \
  --save-best-config examples/workflow_config_gcn_Alpha360_best.yaml
```

The tuning script reuses the Qlib dataset and workflow config, samples model hyperparameters, maximizes validation IC, and can write both a JSON summary and a best-parameter YAML.

## Verified Run

A full GPU run was completed locally on:

- GPU: NVIDIA GeForce RTX 3070
- device: `cuda:0`
- epochs configured: 200
- early stop: 20

Key validation and backtest results:

| Metric | Value |
| --- | ---: |
| Best valid score | 0.104956 @ epoch 52 |
| IC mean | 0.0494 |
| Rank IC mean | 0.0622 |
| Strategy total return | 93.85% |
| Benchmark total return | 41.84% |
| Annualized excess return after cost | 8.70% |
| Information ratio after cost | 1.2050 |
| Max drawdown after cost | -7.30% |

These numbers are included as a reproducibility reference for the default workflow and historical Qlib CN data snapshot. They should not be interpreted as live trading performance.

## Tests

```bash
python -m unittest discover -s tests
python scripts/tune_optuna.py --help
```

The unit tests focus on model shape behavior, daily graph batching, and a small fake-dataset training path. They are intentionally lightweight so the public CI does not need to download market data.

## License

Apache License 2.0.
