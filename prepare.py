"""
One-time data preparation for flood susceptibility experiments.
Downloads terrain features from CloudFront COGs and EA surface water labels from WMS.

Usage:
    python prepare.py                  # full prep (default tiles)
    python prepare.py --num-tiles 10   # fewer tiles (for testing)

Data is stored in ~/.cache/autoresearch-flood/.
"""

import os
import sys
import io
import time
import json
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
import rasterio
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300       # training time budget in seconds (5 minutes)

CACHE_DIR = Path.home() / ".cache" / "autoresearch-flood"
TERRAIN_DIR = CACHE_DIR / "terrain"
LABEL_DIR = CACHE_DIR / "labels"
DATASET_DIR = CACHE_DIR / "dataset"

CLOUDFRONT_BASE = "https://d22hqok9vcmc2j.cloudfront.net/cog"
EA_WMS = "https://environment.data.gov.uk/spatialdata/nafra2-risk-of-flooding-from-surface-water/wms"

# Terrain feature products (all safe — local-kernel, no block boundary issues)
FEATURE_PRODUCTS = ["slope", "twi", "tpi", "curvature", "spi", "conditioned"]
FEATURE_NAMES = ["slope", "twi", "tpi", "curvature", "spi", "elevation"]
N_FEATURES = len(FEATURE_PRODUCTS)

TILE_SIZE_M = 5000      # 5km tiles
PIXEL_SIZE_M = 2        # 2m resolution
TILE_PIXELS = TILE_SIZE_M // PIXEL_SIZE_M  # 2500

# WMS label download at full terrain resolution
WMS_RESOLUTION = TILE_PIXELS  # 2500px = 2m
ALPHA_THRESHOLD = 128   # alpha >= this = flood

SAMPLES_PER_TILE = 20000  # max balanced samples per tile
MIN_FLOOD_PIXELS = 200    # skip tiles with fewer flood pixels than this

# ---------------------------------------------------------------------------
# Tile lists (fixed for reproducibility)
# Diverse England coverage: urban, suburban, rural, flat, hilly
# ---------------------------------------------------------------------------

TRAIN_TILES = [
    # London
    (525000, 175000), (530000, 180000), (535000, 175000),
    (520000, 185000), (540000, 170000), (515000, 180000),
    # Birmingham
    (405000, 280000), (410000, 285000), (400000, 275000),
    # Manchester
    (380000, 395000), (385000, 400000),
    # Leeds
    (425000, 435000), (430000, 430000),
    # Bristol
    (355000, 175000), (360000, 170000),
    # Sheffield / Nottingham
    (435000, 385000), (455000, 340000), (460000, 345000),
    # Cambridge
    (545000, 255000),
    # York
    (460000, 450000),
    # Rural / mixed terrain
    (470000, 300000), (500000, 200000), (350000, 300000),
    (480000, 400000), (560000, 250000), (300000, 150000),
    (410000, 350000), (490000, 340000), (370000, 250000),
    (440000, 200000),
]

VAL_TILES = [
    (620000, 310000),  # Norwich
    (445000, 115000),  # Southampton
    (395000, 220000),  # Cheltenham area
    (425000, 565000),  # Newcastle area
    (290000, 95000),   # Exeter
    (510000, 300000),  # East Midlands
    (330000, 120000),  # Dorset
    (420000, 480000),  # North Yorkshire
    (550000, 350000),  # Suffolk
    (380000, 150000),  # Somerset
]

# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def _tile_name(easting: int, northing: int) -> str:
    return f"E{easting:06d}_N{northing:06d}"


def download_terrain_tile(product: str, easting: int, northing: int) -> Path | None:
    """Download a single terrain COG from CloudFront. Returns local path or None."""
    name = _tile_name(easting, northing)
    local_path = TERRAIN_DIR / product / f"{name}.tif"
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{CLOUDFRONT_BASE}/{product}/{name}.tif"

    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "autoresearch-flood/1.0"})
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
            tmp = local_path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(local_path)
            return local_path
        except (URLError, HTTPError, OSError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def download_label_tile(easting: int, northing: int) -> Path | None:
    """Download EA surface water label from WMS as binary numpy array."""
    name = _tile_name(easting, northing)
    local_path = LABEL_DIR / f"{name}.npy"
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    xmin, ymin = easting, northing
    xmax, ymax = easting + TILE_SIZE_M, northing + TILE_SIZE_M
    params = (
        f"SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        f"&LAYERS=rofsw&FORMAT=image/png&TRANSPARENT=true"
        f"&SRS=EPSG:27700"
        f"&BBOX={xmin},{ymin},{xmax},{ymax}"
        f"&WIDTH={WMS_RESOLUTION}&HEIGHT={WMS_RESOLUTION}"
    )
    url = f"{EA_WMS}?{params}"

    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "autoresearch-flood/1.0"})
            with urlopen(req, timeout=120) as resp:
                png_data = resp.read()

            img = Image.open(io.BytesIO(png_data))
            arr = np.array(img)

            if arr.ndim == 3 and arr.shape[2] == 4:
                # RGBA — threshold alpha channel
                label = (arr[:, :, 3] >= ALPHA_THRESHOLD).astype(np.uint8)
            elif arr.ndim == 3:
                # RGB — treat dark pixels as flood
                label = (arr.mean(axis=2) < 200).astype(np.uint8)
            else:
                label = np.zeros((WMS_RESOLUTION, WMS_RESOLUTION), dtype=np.uint8)

            np.save(local_path, label)
            return local_path
        except (URLError, HTTPError, OSError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def download_all_data(tiles: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Download all tile data. Returns list of tiles where all downloads succeeded."""
    print(f"Downloading data for {len(tiles)} tiles...")
    valid = []
    for i, (e, n) in enumerate(tiles):
        name = _tile_name(e, n)
        print(f"  [{i+1}/{len(tiles)}] {name}...", end=" ", flush=True)

        ok = True
        for product in FEATURE_PRODUCTS:
            if download_terrain_tile(product, e, n) is None:
                ok = False
                break
        if ok and download_label_tile(e, n) is None:
            ok = False

        if ok:
            valid.append((e, n))
            print("OK")
        else:
            print("SKIP")

    print(f"Downloaded {len(valid)}/{len(tiles)} tiles successfully")
    return valid


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def load_tile_features(easting: int, northing: int) -> np.ndarray | None:
    """Load all feature bands for a tile. Returns (H, W, N_FEATURES) float32 or None."""
    name = _tile_name(easting, northing)
    bands = []
    for product in FEATURE_PRODUCTS:
        path = TERRAIN_DIR / product / f"{name}.tif"
        if not path.exists():
            return None
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
        # Replace nodata sentinel values with NaN
        if nodata is not None:
            data[data == nodata] = np.nan
        # Also catch common sentinel values
        data[data <= -9999] = np.nan
        bands.append(data)
    return np.stack(bands, axis=-1)


def load_tile_labels(easting: int, northing: int) -> np.ndarray | None:
    """Load label raster for a tile. Returns (H, W) uint8 or None."""
    name = _tile_name(easting, northing)
    path = LABEL_DIR / f"{name}.npy"
    if not path.exists():
        return None
    return np.load(path)


def sample_tile(easting: int, northing: int, n_samples: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sample balanced flood/non-flood pixels from one tile.
    Returns (X, y) with X shape (n, N_FEATURES) and y shape (n,), or None.
    """
    features = load_tile_features(easting, northing)
    labels = load_tile_labels(easting, northing)
    if features is None or labels is None:
        return None

    feat_h, feat_w = features.shape[:2]
    lab_h, lab_w = labels.shape

    # Upscale labels to match feature resolution if needed
    if (lab_h, lab_w) != (feat_h, feat_w):
        label_img = Image.fromarray(labels, mode='L')
        label_img = label_img.resize((feat_w, feat_h), Image.NEAREST)
        labels = np.array(label_img)

    # Valid mask: all features must be finite
    valid = np.all(np.isfinite(features), axis=-1)

    flood_idx = np.argwhere(valid & (labels == 1))
    nonflood_idx = np.argwhere(valid & (labels == 0))

    if len(flood_idx) < MIN_FLOOD_PIXELS:
        return None

    # Balanced sampling: equal flood and non-flood
    n_each = min(n_samples // 2, len(flood_idx), len(nonflood_idx))
    if n_each < MIN_FLOOD_PIXELS // 2:
        return None

    flood_sample = flood_idx[rng.choice(len(flood_idx), n_each, replace=False)]
    nonflood_sample = nonflood_idx[rng.choice(len(nonflood_idx), n_each, replace=False)]

    all_idx = np.concatenate([flood_sample, nonflood_sample])
    rows, cols = all_idx[:, 0], all_idx[:, 1]

    X = features[rows, cols]  # (2*n_each, N_FEATURES)
    y = labels[rows, cols]    # (2*n_each,)

    return X, y


def create_dataset(tile_list: list[tuple[int, int]], n_per_tile: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create dataset by sampling pixels from tiles."""
    rng = np.random.default_rng(seed)

    all_X, all_y = [], []
    for e, n in tile_list:
        name = _tile_name(e, n)
        result = sample_tile(e, n, n_per_tile, rng)
        if result is None:
            print(f"  Skipping {name} (insufficient flood pixels or data)")
            continue
        X, y = result
        all_X.append(X)
        all_y.append(y)
        print(f"  {name}: {len(X):,} pixels ({y.mean():.1%} flood)")

    if not all_X:
        print("ERROR: No valid tiles found!")
        sys.exit(1)

    return np.concatenate(all_X), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_auc(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    ROC AUC on validation set. This is the ground truth metric.
    Higher is better (1.0 = perfect, 0.5 = random).
    """
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for flood susceptibility experiments"
    )
    parser.add_argument(
        "--num-tiles", type=int, default=-1,
        help="Number of training tiles (-1 = all)"
    )
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Download training data
    train_tiles = TRAIN_TILES[:args.num_tiles] if args.num_tiles > 0 else TRAIN_TILES
    print("=== Training tiles ===")
    valid_train = download_all_data(train_tiles)
    print()

    # Download validation data
    print("=== Validation tiles ===")
    valid_val = download_all_data(VAL_TILES)
    print()

    if not valid_train or not valid_val:
        print("ERROR: Need at least 1 valid train tile and 1 valid val tile.")
        sys.exit(1)

    # Create datasets
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Creating training dataset ===")
    X_train, y_train = create_dataset(valid_train, SAMPLES_PER_TILE, seed=42)
    print(f"Training set: {len(X_train):,} pixels, {y_train.mean():.1%} flood")
    print()

    print("=== Creating validation dataset ===")
    X_val, y_val = create_dataset(valid_val, SAMPLES_PER_TILE, seed=123)
    print(f"Validation set: {len(X_val):,} pixels, {y_val.mean():.1%} flood")
    print()

    # Save arrays
    np.save(DATASET_DIR / "X_train.npy", X_train)
    np.save(DATASET_DIR / "y_train.npy", y_train)
    np.save(DATASET_DIR / "X_val.npy", X_val)
    np.save(DATASET_DIR / "y_val.npy", y_val)

    # Save metadata
    meta = {
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "flood_rate_train": float(y_train.mean()),
        "flood_rate_val": float(y_val.mean()),
        "train_tiles": [[int(e), int(n)] for e, n in valid_train],
        "val_tiles": [[int(e), int(n)] for e, n in valid_val],
        "samples_per_tile": SAMPLES_PER_TILE,
    }
    with open(DATASET_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {DATASET_DIR}")
    print()
    print("Done! Ready to train.")
