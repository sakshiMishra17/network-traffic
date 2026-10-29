"""Shared helpers: json io, feature transform, loading processed data, metrics."""
import json
import time

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src import config


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def signed_log1p(x):
    """sign(x) * log(1 + |x|). Tames the huge ranges of flow features (bytes/s,
    durations in microseconds) and keeps negative values (e.g. Init_Win_bytes = -1).
    Used identically in training and in prediction."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def load_processed():
    """Load the arrays + class names written by src.preprocess."""
    path = config.DATA_PROCESSED / "data.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.preprocess` first.")
    d = np.load(path, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    data["class_names"] = load_json(config.MODELS_DIR / "class_names.json")
    return data


def evaluate_and_save(name, kind, model_file, predict_fn, data, train_time, extra=None):
    """Compute metrics on val + test, print a summary and write results/metrics_<name>.json.

    predict_fn(X) must return integer class indices.
    """
    class_names = data["class_names"]
    labels = list(range(len(class_names)))

    val_pred = predict_fn(data["X_val"])
    t0 = time.perf_counter()
    test_pred = predict_fn(data["X_test"])
    infer_s = time.perf_counter() - t0
    y_test = data["y_test"]

    report = classification_report(y_test, test_pred, labels=labels,
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    metrics = {
        "model": name,
        "kind": kind,
        "model_file": model_file,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
        "val_macro_f1": float(f1_score(data["y_val"], val_pred, average="macro", zero_division=0)),
        "train_time_s": float(train_time),
        "infer_ms_per_1000": float(infer_s * 1000.0 / max(len(y_test) / 1000.0, 1e-9)),
        "per_class": {c: report[c] for c in class_names},
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=labels).tolist(),
        "class_names": class_names,
    }
    if extra:
        metrics.update(extra)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(metrics, config.RESULTS_DIR / f"metrics_{name}.json")
    print(f"[{name}] test acc={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}  "
          f"train={metrics['train_time_s']:.1f}s  infer={metrics['infer_ms_per_1000']:.1f} ms/1000 rows")
    return metrics
