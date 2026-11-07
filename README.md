# Underwater Tunnel Route Optimization

## Current implementation status

Phase 1 is underway. The repository contains the GEBCO preprocessing module and a
configuration file for the current Strait of Gibraltar working crop.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the data pipeline

From the repository root:

```bash
python -m src.data --config config.yaml
```

The script reads `data/raw/gebco_gibraltar_initial.nc` and writes:

- `data/processed/gibraltar_metric_grid.npz`
- `data/processed/metadata.json`

The NPZ contains `x`, `y`, `elevation`, `depth`, and `water_mask`.

## Current modelling choices

- Dataset: GEBCO_2026.
- Working crop: longitude [-5.65, -5.25], latitude [35.92, 36.07].
- Projected CRS: UTM Zone 30N (EPSG:32630).
- Metric grid spacing: 400 m.
- Resampling: nearest-neighbour source-cell sampling.
- Endpoints A and B are provisional and are recorded in `config.yaml`.

The working crop and endpoints must be checked before they are frozen as the final
study region. The depth array is positive downward and denotes seafloor depth,
not tunnel depth below the seabed.

## Reproducibility

The raw GEBCO subset is retained under `data/raw/`. Do not overwrite it.
All modelling parameters should be stored in `config.yaml` or in per-run result
metadata. Do not report results that have not been produced by the code.
