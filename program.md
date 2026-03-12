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

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

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

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_auc	training_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.850000) — use 0.000000 for crashes
3. training_seconds (e.g. 45.2) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_auc	training_seconds	status	description
a1b2c3d	0.850000	45.2	keep	baseline random forest
b2c3d4e	0.862000	52.1	keep	increase n_estimators to 500
c3d4e5f	0.848000	120.5	discard	gradient boosting (slower + worse)
d4e5f6g	0.000000	0.0	crash	bug in feature engineering
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_auc:\|^training_seconds:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_auc improved (higher), you "advance" the branch, keeping the git commit
9. If val_auc is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take well under 5 minutes. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (a bug, wrong API, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical approaches, try feature engineering you haven't attempted yet. The loop runs until the human interrupts you, period.

## Ideas to explore

Here are starting points for experimentation (ordered roughly by expected impact):

**Model types** (all in scikit-learn):
- HistGradientBoostingClassifier — fast gradient boosting, often best for tabular
- GradientBoostingClassifier — slower but sometimes better
- ExtraTreesClassifier — more random splits, can be surprisingly good
- AdaBoostClassifier, BaggingClassifier — ensemble wrappers
- VotingClassifier, StackingClassifier — combine multiple models

**Feature engineering** (modify train.py to create derived features):
- Log transforms: `np.log1p(slope)`, `np.log1p(spi)`
- Interactions: `slope * twi`, `tpi * curvature`
- Polynomial features from sklearn.preprocessing
- Binning / discretization
- Rank transforms

**Hyperparameter tuning**:
- n_estimators (more trees = slower but potentially better)
- max_depth (deeper = more complex, risk of overfitting)
- min_samples_leaf, min_samples_split
- max_features (fraction of features per split)
- class_weight (even though data is balanced, weighting can help)

**Preprocessing**:
- StandardScaler, RobustScaler
- Clipping outliers
- Missing value strategies (though data should be clean)

**Domain knowledge hints**:
- TWI is the most physically meaningful predictor — water flows downhill and accumulates
- Slope and curvature together capture micro-topography
- TPI captures whether a pixel is in a local depression (negative = depression)
- SPI captures erosive potential of flowing water
- Elevation alone is less predictive than relative measures (TWI, TPI)
