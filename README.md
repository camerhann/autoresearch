# autoresearch-flood

Autonomous flood susceptibility research. Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted for geospatial ML.

Give an AI agent a flood susceptibility training setup and let it experiment autonomously. It modifies the model code, trains, checks if the result improved, keeps or discards, and repeats. You come back to a log of experiments and a better model.

## How it works

Three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads terrain COGs + EA flood labels), sampling, evaluation. Not modified by the agent.
- **`train.py`** — the single file the agent edits. Model type, hyperparameters, feature engineering, preprocessing. **This file is edited and iterated on by the agent**.
- **`program.md`** — instructions for the agent. Point your agent here and let it go.

**Features** (2m resolution, England-wide):
- Slope, TWI, TPI, curvature, SPI, elevation (conditioned DEM)
- All from [Wrench](https://wrench.build) national terrain products on S3

**Labels**: EA Risk of Flooding from Surface Water (RoFSW) — binary flood extent from WMS.

**Metric**: ROC AUC on spatially held-out validation tiles. Higher is better.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU needed (scikit-learn, CPU-only).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Download data and create training/validation datasets (~5-10 min)
uv run prepare.py

# 3. Run a single training experiment (~seconds)
uv run train.py
```

## Running the agent

Spin up Claude Code (or similar) in this repo, then prompt:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

## Project structure

```
prepare.py      — constants, data prep + evaluation (do not modify)
train.py        — model + training (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies (scikit-learn, rasterio, numpy, etc.)
```

## Data sources

- **Terrain features**: CloudFront-served COGs from `s3://aegaea-lidar/cog/` (5km tiles, 2m resolution, EPSG:27700)
- **Flood labels**: EA NaFRA2 Risk of Flooding from Surface Water via WMS (rasterized at 2m)

## License

MIT
