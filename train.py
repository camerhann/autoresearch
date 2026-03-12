"""
Flood susceptibility model — the experiment file.

Modify this file to improve val_auc. Everything is fair game:
model type, hyperparameters, feature engineering, preprocessing.

Run: uv run train.py
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

from prepare import CACHE_DIR, FEATURE_NAMES, N_FEATURES, TIME_BUDGET, evaluate_auc

DATASET_DIR = CACHE_DIR / "dataset"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

X_train = np.load(DATASET_DIR / "X_train.npy")
y_train = np.load(DATASET_DIR / "y_train.npy")
X_val = np.load(DATASET_DIR / "X_val.npy")
y_val = np.load(DATASET_DIR / "y_val.npy")

print(f"Train: {len(X_train):,} samples, {N_FEATURES} features, {y_train.mean():.1%} flood")
print(f"Val:   {len(X_val):,} samples, {y_val.mean():.1%} flood")

# ---------------------------------------------------------------------------
# Feature engineering — help HGBT with explicit interactions
# Features: slope(0), twi(1), tpi(2), curvature(3), spi(4), elevation(5)
# ---------------------------------------------------------------------------

def add_features(X):
    slope = X[:, 0]
    twi = X[:, 1]
    tpi = X[:, 2]
    curv = X[:, 3]
    spi = X[:, 4]
    elev = X[:, 5]
    new = np.column_stack([
        X,
        tpi * twi,              # depression + wetness
        slope * twi,            # steep + wet
        tpi * curv,             # depression + concavity
        np.log1p(np.abs(spi)),  # log SPI (skewed)
        tpi * slope,            # depression + gradient
        twi * curv,             # wetness + curvature
        twi / (slope + 0.01),   # wetness per unit slope (flood accumulation)
    ])
    return new

X_train = add_features(X_train)
X_val = add_features(X_val)

print(f"Engineered features: {X_train.shape[1]} total")

# ---------------------------------------------------------------------------
# Model — Depth-spectrum voting ensemble: RF + ET at varied depths
# ---------------------------------------------------------------------------

DEPTHS = [2, 3, 5, 7, 10, 12, 15, 20, 25]
MIN_LEAF = {2: 200, 3: 100, 5: 50, 7: 40, 10: 30, 12: 25, 15: 20, 20: 10, 25: 5}
TREES = 300

estimators = []
for i, d in enumerate(DEPTHS):
    rf = RandomForestClassifier(
        n_estimators=TREES, max_depth=d, min_samples_leaf=MIN_LEAF[d],
        n_jobs=-1, random_state=42 + i,
    )
    et = ExtraTreesClassifier(
        n_estimators=TREES, max_depth=d, min_samples_leaf=MIN_LEAF[d],
        n_jobs=-1, random_state=42 + i,
    )
    estimators.append((f"rf_d{d}", rf))
    estimators.append((f"et_d{d}", et))

model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

t0 = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t0

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

val_auc = evaluate_auc(model, X_val, y_val)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print("---")
print(f"val_auc:          {val_auc:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {time.time() - t0:.1f}")
print(f"n_features:       {X_train.shape[1]}")
print(f"n_train:          {len(X_train)}")
print(f"n_val:            {len(X_val)}")
print(f"flood_rate_train: {y_train.mean():.4f}")
print(f"flood_rate_val:   {y_val.mean():.4f}")
print(f"model_type:       VotingClassifier(RF+ET depth spectrum)+feat_eng")
