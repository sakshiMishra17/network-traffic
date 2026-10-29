"""Inference helpers used by the dashboard and the command line.

CLI:  python -m src.predict data/sample/demo_test.csv --model RandomForest --out preds.csv
"""
import argparse
import glob

import joblib
import numpy as np
import pandas as pd

from src import config
from src.utils import load_json, signed_log1p


def load_artifacts():
    """Scaler + class names + feature order + train medians saved by preprocess."""
    needed = ["scaler.pkl", "class_names.json", "feature_names.json", "medians.json"]
    missing = [n for n in needed if not (config.MODELS_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {config.MODELS_DIR}. "
                                f"Run `python -m src.preprocess` first.")
    return {
        "scaler": joblib.load(config.MODELS_DIR / "scaler.pkl"),
        "class_names": load_json(config.MODELS_DIR / "class_names.json"),
        "feature_names": load_json(config.MODELS_DIR / "feature_names.json"),
        "medians": np.asarray(load_json(config.MODELS_DIR / "medians.json"), dtype=np.float64),
    }


def build_matrix(df, art):
    """Raw flow dataframe -> scaled numpy matrix, using exactly the training recipe.
    Extra columns (Label, IPs, ...) are ignored; inf/NaN are replaced by train medians
    so every uploaded row still gets a prediction."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    feats = art["feature_names"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} required column(s) missing, e.g. {missing[:5]}. "
                         f"The CSV must be in CIC-IDS2017 flow format.")
    # copy=True: pandas 3 may return a read-only view, and we modify X below
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    X[~np.isfinite(X)] = np.nan
    X = np.where(np.isnan(X), art["medians"][None, :], X)
    return art["scaler"].transform(signed_log1p(X))


def list_models():
    """Models that have a metrics file AND a model file, best macro-F1 first."""
    found = []
    for path in glob.glob(str(config.RESULTS_DIR / "metrics_*.json")):
        m = load_json(path)
        if (config.MODELS_DIR / m["model_file"]).exists():
            found.append({"name": m["model"], "kind": m["kind"], "file": m["model_file"],
                          "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
    return sorted(found, key=lambda m: -m["macro_f1"])


def load_model(name):
    info = {m["name"]: m for m in list_models()}
    if name not in info:
        raise ValueError(f"Model '{name}' not found. Available: {list(info)}")
    path = config.MODELS_DIR / info[name]["file"]
    if info[name]["kind"] == "torch":
        from src.dl_models import TorchPredictor   # needs torch installed
        return TorchPredictor(path)
    return joblib.load(path)


def predict_dataframe(df, model, art):
    """Returns a dataframe with Prediction and Confidence (max class probability)."""
    X = build_matrix(df, art)
    proba = model.predict_proba(X)
    idx = proba.argmax(axis=1)
    names = np.asarray(art["class_names"])
    out = pd.DataFrame({"Prediction": names[idx], "Confidence": proba.max(axis=1)},
                       index=df.index)
    return out


def read_csv_flexible(path_or_buffer):
    """Read a CIC-IDS2017 style CSV (latin-1 never fails on the odd label character)."""
    return pd.read_csv(path_or_buffer, encoding="latin-1", low_memory=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--model", default=None, help="model name (default: best macro-F1)")
    ap.add_argument("--out", default=None, help="write predictions to this csv")
    a = ap.parse_args()

    models = list_models()
    if not models:
        raise SystemExit("No trained models found. Run train_ml / train_dl first.")
    name = a.model or models[0]["name"]
    art = load_artifacts()
    raw = read_csv_flexible(a.csv)
    pred = predict_dataframe(raw, load_model(name), art)
    print(f"Model: {name}   rows: {len(pred)}")
    print(pred["Prediction"].value_counts().to_string())
    if a.out:
        pd.concat([raw.reset_index(drop=True), pred.reset_index(drop=True)], axis=1).to_csv(a.out, index=False)
        print("saved", a.out)
