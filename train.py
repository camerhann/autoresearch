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

print(f"Train: {len(X_train):,} samples, {N_FEATURES} features, {y_train.mean():.1%} flood")
print(f"Val:   {len(X_val):,} samples, {y_val.mean():.1%} flood")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
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
feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])

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
