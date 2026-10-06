# Pytorch Lightning Forecasting Repo

Minimal working copy of the probabilistic forecasting framework used in the paper. Built on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## Structure

```
.
├── main_lightning.py          # Entry point: parses args, builds dataset, selects model, runs trainer
├── update_data.sh             # Downloads latest raw data from the continuously updated dataset
├── scripts/
│   ├── run_ANGEL.sh           # Hyperparameter spec reported in the paper
│   └── run_*.sh               # Scripts for benchmarks and ablations
├── models/
│   ├── common.py              # Baseclass_forecast (L.LightningModule): training loop, losses, logging, test diagnostics
│   ├── ANGEL.py               # Our model (nn.Module architecture + Lightning wrapper)
│   ├── iTransformer.py        # iTransformer baseline
│   ├── TimeXer.py             # TimeXer baseline
│   ├── TSMixer.py             # TSMixer baseline
│   ├── CNN.py                 # CNN baseline (early-warning)
│   ├── TiDE.py                # TiDE baseline
│   ├── NBEATS.py              # N-BEATS baseline
│   ├── custom.py              # Custom/experimental architectures
│   ├── helper_classes.py      # Shared building blocks (RevIN, quantile heads, spliced GPD, etc.)
│   ├── losses.py              # Loss functions (CRPS, twCRPS, focal, Gaussian variants)
│   └── utils.py               # Evaluation utilities and plotting helpers
├── layers/                    # Reusable nn.Module pieces (attention, embeddings, encoder/decoder blocks)
├── data_loader/
│   ├── DataModules.py         # Lightning DataModules (forecast and early-warning)
│   └── Datasets.py            # PyTorch Datasets
└── utils/
    ├── argument_parser.py     # CLI argument definitions
    ├── build_dataset.py       # Feature engineering and dataset construction
    └── losses.py              # Standalone loss utilities (pinball, expectile)
```

## Design

Each model file in `models/` defines a `nn.Module` architecture and a thin Lightning wrapper subclassing `Baseclass_forecast` from `models/common.py`. All shared training logic lives in `Baseclass_forecast`: optimiser, LR scheduling, loss dispatch, MLflow logging, and test-time diagnostics (PIT histograms, VaR/ES calibration, attention visualisations, etc.).

Data loading is handled by `data_loader/DataModules.py` (Lightning `DataModule`) and `data_loader/Datasets.py` (PyTorch `Dataset`). Feature construction lives in `utils/build_dataset.py`.

## Our Model: ANGEL

ANGEL is our proposed architecture (`models/ANGEL.py`). The paper-reported hyperparameters are in `scripts/run_ANGEL.sh`. Run it from the repo root:

```bash
bash scripts/run_ANGEL.sh
```

## Baselines

Other `run_*.sh` scripts under `scripts/` reproduce the benchmark runs.

## Setup

### 1. Environment

```bash
pip install -r requirements.txt
```

### 2. MLflow logging

Copy `.env.example` to `.env` and fill in your MLflow tracking URI:

```bash
cp .env.example .env
# edit .env and set MLFLOW_TRACKING_URI=<your URI>
```

If you do not have an MLflow server, remove the `--remote_logging` flag from the relevant script before running.

### 3. Data

Download the latest raw data:

```bash
bash update_data.sh
```

This pulls all data archives from the [`MSCA-DN-Digital-Finance/stablecoin-onchain-data`](https://github.com/MSCA-DN-Digital-Finance/stablecoin-onchain-data) GitHub release and extracts them locally.
