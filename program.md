# autoresearch-flood

Autonomous flood susceptibility research. Train ML models to predict pluvial (surface water) flood risk from terrain features.

## Context

**What we're building**: A per-pixel flood susceptibility score for England, trained on 2m terrain features with EA surface water flood maps as ground truth. Analogous to 7analytics' pluvial flood index but trained on gridded model outputs, not claims data.

**Features** (all at 2m resolution, England-wide terrain products):
- `slope` — terrain gradient in degrees
- `twi` — topographic wetness index (log(a/tan(b)), higher = wetter)
- `tpi` — topographic position index (elevation relative to neighbours)
- `curvature` — profile curvature (concave vs convex)
- `spi` — stream power index
- `elevation` — hydrologically conditioned DEM (metres, BNG)

**Labels**: Binary — EA Risk of Flooding from Surface Water (RoFSW). 1 = flood extent, 0 = no flood. Training data is balanced 50/50.

**Metric**: ROC AUC on spatially held-out validation tiles. Higher is better (1.0 = perfect, 0.5 = random).

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data download, sampling, evaluation. Do not modify.
   - `train.py` — the file you modify. Model, hyperparameters, feature engineering.
4. **Verify data exists**: Check that `~/.cache/autoresearch-flood/dataset/` contains X_train.npy, y_train.npy, X_val.npy, y_val.npy. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CPU. The training script should finish within a **5 minute** time budget. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model type, hyperparameters, feature engineering, preprocessing, ensemble methods.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_auc` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_auc.** Since the time budget is fixed at 5 minutes, everything is fair game: change the model, the hyperparameters, add feature engineering, try different classifiers.

**Available libraries**: scikit-learn, numpy, pandas, matplotlib. These are powerful — scikit-learn has dozens of classifiers, preprocessors, and pipeline tools.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_auc:          0.850000
training_seconds: 45.2
total_seconds:    46.1
n_features:       6
n_train:          400000
n_val:            100000
flood_rate_train: 0.5000
flood_rate_val:   0.5000
n_estimators:     200

Feature importance:
  slope           0.2345
  twi             0.1890
  ...
```

You can extract the key metric from the log file:

```
grep "^val_auc:" run.log
```

## Logging and Traceability

**CRITICAL**: Every experiment must be fully traceable. This research runs autonomously and the human needs to reconstruct what happened after the fact.

### Per-experiment logs

After each experiment, save the full log to `logs/`:

```bash
cp run.log logs/$(git rev-parse --short HEAD).log
```

This preserves the complete output for every experiment, linked by commit hash.

### results.tsv

Tab-separated, 5 columns. **Committed to git** — this is the primary research artifact.

```
commit	val_auc	training_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.850000) — use 0.000000 for crashes
3. training_seconds (e.g. 45.2) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

### Vault progress reports

Every **10 experiments** (or on a significant milestone), write a progress report to:

```
/Users/camerhann/ChrisVault/Donna/Research/flood-susceptibility/YYYY-MM-DD-progress.md
```

The report should include:
- Current best val_auc and which model achieved it
- Table of all experiments so far (from results.tsv)
- Top feature importances from best model
- Key insights or surprises
- Next directions to explore

Also write a **final report** on the last experiment before stopping.

### Git backups

Every **5 experiments**, push the branch to origin:

```bash
git push origin autoresearch/<tag>
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code
3. git commit the code change
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_auc:\|^training_seconds:" run.log`
6. If empty, crashed. Run `tail -n 50 run.log` for traceback. Fix if trivial, skip if not.
7. Save log: `cp run.log logs/$(git rev-parse --short HEAD).log`
8. Append results to results.tsv
9. `git add results.tsv logs/ && git commit -m "results: {description}"`
10. If val_auc improved (higher): keep — advance the branch
11. If val_auc equal or worse: discard — `git revert` the code change (keep the results commit so the record is preserved)
12. Every 5 experiments: `git push origin autoresearch/<tag>`
13. Every 10 experiments: write vault progress report

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Fix trivial bugs and re-run. If fundamentally broken, log "crash" and move on.

**NEVER STOP**: The loop runs until the human interrupts you. If you run out of ideas, think harder — combine near-misses, try radical approaches, read scikit-learn docs for unexplored models.

## Ideas to explore

**Model types** (all in scikit-learn):
- HistGradientBoostingClassifier — fast gradient boosting, often best for tabular
- GradientBoostingClassifier — slower but sometimes better
- ExtraTreesClassifier — more random splits, can be surprisingly good
- AdaBoostClassifier, BaggingClassifier — ensemble wrappers
- VotingClassifier, StackingClassifier — combine multiple models

**Feature engineering** (create derived features in train.py):
- Log transforms: `np.log1p(slope)`, `np.log1p(spi)`
- Interactions: `slope * twi`, `tpi * curvature`
- Polynomial features from sklearn.preprocessing
- Binning / discretization
- Rank transforms

**Hyperparameter tuning**:
- n_estimators, max_depth, min_samples_leaf, max_features
- learning_rate (for boosting)
- class_weight

**Domain knowledge hints**:
- TWI is the most physically meaningful predictor — water flows downhill and accumulates
- TPI captures local depressions (negative TPI = depression = flood prone)
- Slope and curvature together capture micro-topography
- SPI captures erosive potential of flowing water
- Elevation alone is less predictive than relative measures (TWI, TPI)
- Interactions like slope x TWI encode "steep but high drainage area" vs "flat with low drainage"
