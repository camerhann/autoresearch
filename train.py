"""
Flood susceptibility model — the experiment file.

Modify this file to improve val_auc. Everything is fair game:
model type, hyperparameters, feature engineering, preprocessing.

Run: uv run train.py
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

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
# Model — Voting ensemble: RF + HGBT + ET
# ---------------------------------------------------------------------------

rf1 = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

rf2 = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=99,
)

hgbt = HistGradientBoostingClassifier(
    max_iter=1000,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=50,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)

et = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

rf3 = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=30,
    n_jobs=-1,
    random_state=77,
)

et2 = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=30,
    n_jobs=-1,
    random_state=77,
)

et3 = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=55,
)

rf4 = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=33,
)

et4 = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=33,
)

rf5 = RandomForestClassifier(
    n_estimators=500,
    max_depth=7,
    min_samples_leaf=40,
    n_jobs=-1,
    random_state=11,
)

rf6 = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=22,
)

et5 = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=7,
    min_samples_leaf=40,
    n_jobs=-1,
    random_state=11,
)

et6 = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=22,
)

rf7 = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=25,
    n_jobs=-1,
    random_state=44,
)

et7 = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=25,
    n_jobs=-1,
    random_state=44,
)

model = VotingClassifier(
    estimators=[
        ("rf1", rf1), ("rf2", rf2), ("rf3", rf3), ("rf4", rf4), ("rf5", rf5), ("rf6", rf6), ("rf7", rf7),
        ("hgbt", hgbt),
        ("et", et), ("et2", et2), ("et3", et3), ("et4", et4), ("et5", et5), ("et6", et6), ("et7", et7),
    ],
    voting="soft",
    n_jobs=-1,
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
print(f"model_type:       VotingClassifier(RF+HGBT+ET)+feat_eng")
