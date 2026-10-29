"""Step 1: clean raw CIC-IDS2017 CSVs, map labels, split, scale and save.

Usage:  python -m src.preprocess [--raw data/raw] [--max-per-class 30000]

Outputs
  data/processed/data.npz          X_train/X_val/X_test (scaled), y_*
  models/scaler.pkl                StandardScaler (fit on TRAIN only)
  models/label_encoder.pkl         LabelEncoder
  models/class_names.json          class order used everywhere
  models/feature_names.json        feature column order used everywhere
  models/medians.json              train medians (to fill NaN/inf at prediction time)
  results/class_counts.json        rows per class after cleaning
  data/sample/demo_test.csv        raw-format test rows for the dashboard demo
"""
import argparse
import glob
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config
from src.config import LABEL_COL, RANDOM_STATE
from src.utils import save_json, signed_log1p


# ----------------------------------------------------------------- labels
def normalize_label(raw):
    """'DoS Hulk ' -> 'dos hulk'; strips spaces and odd characters (\\x96 etc.)."""
    s = str(raw).strip().lower()
    return re.sub(r"[^a-z0-9\- ]", "", s).strip()


def map_label(raw):
    """Map a raw CIC-IDS2017 label to one of the project classes, or None to drop it."""
    s = normalize_label(raw)
    if s in config.CLASS_MAP_EXACT:
        return config.CLASS_MAP_EXACT[s]
    for prefix, cls in config.CLASS_MAP_PREFIX.items():
        if s == prefix or s.startswith(prefix + " "):
            return cls
    return None


# --------------------------------------------------------------- cleaning
def clean_frame(df):
    """Clean ONE raw dataframe: strip names, drop id columns, remove inf/NaN,
    map labels, drop duplicates. Returns features (float64) + Label column."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if LABEL_COL not in df.columns:
        raise ValueError(f"Column '{LABEL_COL}' not found. Columns: {list(df.columns)[:10]}...")
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df[LABEL_COL] = df[LABEL_COL].map(map_label)
    df = df.dropna(subset=[LABEL_COL])
    df = df.drop_duplicates()
    df[feature_cols] = df[feature_cols].astype(np.float64)
    return df.reset_index(drop=True)


def cap_per_class(df, cap, seed=RANDOM_STATE):
    parts = [g.sample(n=min(len(g), cap), random_state=seed) for _, g in df.groupby(LABEL_COL)]
    return pd.concat(parts, ignore_index=True)


def load_and_clean(raw_dir=None, max_per_class=None, verbose=True):
    """Read every CSV in raw_dir one by one (memory friendly), clean, cap per class."""
    raw_dir = Path(raw_dir or config.DATA_RAW)
    max_per_class = max_per_class or config.MAX_PER_CLASS
    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. Put the CIC-IDS2017 CSVs there, or create demo "
            f"data with `python -m src.generate_sample_data`.")

    frames, columns = [], None
    for f in files:
        # latin-1 never fails; real CIC files contain a non-UTF8 character in a label
        raw = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df = clean_frame(raw)
        if df.empty:
            if verbose:
                print(f"  {Path(f).name}: 0 usable rows (skipped)")
            continue
        if columns is None:
            columns = list(df.columns)
        elif set(df.columns) != set(columns):
            diff = set(df.columns) ^ set(columns)
            raise ValueError(f"{Path(f).name} has different columns than the first file: {sorted(diff)[:5]}")
        df = cap_per_class(df[columns], max_per_class)
        if verbose:
            print(f"  {Path(f).name}: kept {len(df)} rows {df[LABEL_COL].value_counts().to_dict()}")
        frames.append(df)

    if not frames:
        raise ValueError("No usable rows found in any file (labels not recognised?).")
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df = cap_per_class(df, max_per_class)
    return df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


# ------------------------------------------------------- split + scaling
def split_and_scale(df, verbose=True):
    counts = df[LABEL_COL].value_counts()
    small = counts[counts < config.MIN_SAMPLES_PER_CLASS].index.tolist()
    if small:
        print(f"WARNING: dropping classes with < {config.MIN_SAMPLES_PER_CLASS} rows: {small}")
        df = df[~df[LABEL_COL].isin(small)].reset_index(drop=True)
    if df[LABEL_COL].nunique() < 2:
        raise ValueError("Need at least 2 classes after cleaning.")

    feature_names = [c for c in df.columns if c != LABEL_COL]
    X_raw = df[feature_names].to_numpy(dtype=np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])

    # stratified 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)

    medians = np.median(X_tr, axis=0)                      # train only
    scaler = StandardScaler().fit(signed_log1p(X_tr))      # fit on TRAIN only (no leakage)
    out = {
        "X_train": scaler.transform(signed_log1p(X_tr)), "y_train": y_tr,
        "X_val": scaler.transform(signed_log1p(X_val)), "y_val": y_val,
        "X_test": scaler.transform(signed_log1p(X_te)), "y_test": y_te,
    }
    if verbose:
        print(f"Split sizes: train={len(y_tr)} val={len(y_val)} test={len(y_te)}")
    return out, le, scaler, feature_names, medians, X_te, y_te


def run_preprocess(raw_dir=None, max_per_class=None, verbose=True):
    config.ensure_dirs()
    if verbose:
        print("Loading and cleaning raw files...")
    df = load_and_clean(raw_dir, max_per_class, verbose)
    class_counts = df[LABEL_COL].value_counts().to_dict()
    if verbose:
        print("Class counts after cleaning:", class_counts)

    arrays, le, scaler, feature_names, medians, X_te_raw, y_te = split_and_scale(df, verbose)

    np.savez_compressed(config.DATA_PROCESSED / "data.npz", **arrays)
    joblib.dump(scaler, config.MODELS_DIR / "scaler.pkl")
    joblib.dump(le, config.MODELS_DIR / "label_encoder.pkl")
    save_json([str(c) for c in le.classes_], config.MODELS_DIR / "class_names.json")
    save_json(feature_names, config.MODELS_DIR / "feature_names.json")
    save_json([float(m) for m in medians], config.MODELS_DIR / "medians.json")
    save_json({k: int(v) for k, v in class_counts.items() if k in set(le.classes_)},
              config.RESULTS_DIR / "class_counts.json")

    # raw-format demo file (unscaled) for the Streamlit dashboard
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_te), size=min(1000, len(y_te)), replace=False)
    demo = pd.DataFrame(X_te_raw[idx], columns=feature_names)
    demo[LABEL_COL] = le.inverse_transform(y_te[idx])
    demo.to_csv(config.DATA_SAMPLE / "demo_test.csv", index=False)

    if verbose:
        print("Classes:", list(le.classes_))
        print("Saved processed data, scaler, encoder and demo file.")
    return {"class_counts": class_counts, "classes": list(le.classes_),
            "n_features": len(feature_names), **{k: len(v) for k, v in arrays.items() if k.startswith("y_")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=None, help="folder with raw CSV files (default data/raw)")
    ap.add_argument("--max-per-class", type=int, default=None, help="cap rows per class")
    args = ap.parse_args()
    run_preprocess(args.raw, args.max_per_class)
