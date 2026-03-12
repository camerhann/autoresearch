"""
Flood susceptibility model — the experiment file.

Modify this file to improve val_auc. Everything is fair game:
model type, hyperparameters, feature engineering, preprocessing.

Run: uv run train.py
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from prepare import CACHE_DIR, FEATURE_NAMES, N_FEATURES, TIME_BUDGET, evaluate_auc

DATASET_DIR = CACHE_DIR / "dataset"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

X_train = np.load(DATASET_DIR / "X_train.npy")
y_train = np.load(DATASET_DIR / "y_train.npy")
X_val = np.load(DATASET_DIR / "X_val.npy")
y_val = np.load(DATASET_DIR / "y_val.npy")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(X):
    slope, twi, tpi, curvature, spi, elev = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    return np.column_stack([
        X,
        slope * twi,           # steep + high drainage area
        tpi * curvature,       # depression shape
        np.log1p(np.abs(spi)), # log stream power
        tpi * twi,             # depression wetness
    ])

X_train = add_features(X_train)
X_val = add_features(X_val)
FEAT_NAMES = FEATURE_NAMES + ["slope_x_twi", "tpi_x_curv", "log_spi", "tpi_x_twi"]

print(f"Train: {len(X_train):,} samples, {X_train.shape[1]} features, {y_train.mean():.1%} flood")
print(f"Val:   {len(X_val):,} samples, {y_val.mean():.1%} flood")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

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

# Feature importance
importances = model.feature_importances_
feat_imp = sorted(zip(FEAT_NAMES, importances), key=lambda x: -x[1])

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
print(f"n_estimators:     {model.n_estimators}")
print()
print("Feature importance:")
for name, imp in feat_imp:
    print(f"  {name:15s} {imp:.4f}")
