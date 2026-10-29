"""Step 2: train classic ML models and save metrics.

Models: RandomForest, HistGradientBoosting (sklearn), XGBoost (only if installed).

Usage:  python -m src.train_ml [--models RandomForest HistGradientBoosting XGBoost]
"""
import argparse
import time

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.config import RANDOM_STATE
from src.utils import evaluate_and_save, load_processed


def build_models(fast=False):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=40 if fast else 150, n_jobs=-1, class_weight="balanced",
            random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=40 if fast else 200, learning_rate=0.1, class_weight="balanced",
            random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=40 if fast else 200, max_depth=8, learning_rate=0.1,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
            eval_metric="mlogloss")
    except ImportError:
        print("xgboost not installed -> skipping XGBoost (pip install xgboost to enable)")
    return models


def main(selected=None):
    config.ensure_dirs()
    data = load_processed()
    models = build_models(config.FAST)
    if selected:
        unknown = [m for m in selected if m not in models]
        if unknown:
            raise SystemExit(f"Unknown/unavailable models: {unknown}. Available: {list(models)}")
        models = {k: v for k, v in models.items() if k in selected}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        t0 = time.perf_counter()
        if name == "XGBoost":  # XGBoost has no class_weight argument -> use sample weights
            w = compute_sample_weight("balanced", data["y_train"])
            model.fit(data["X_train"], data["y_train"], sample_weight=w)
        else:
            model.fit(data["X_train"], data["y_train"])
        train_time = time.perf_counter() - t0

        joblib.dump(model, config.MODELS_DIR / f"{name}.pkl")
        evaluate_and_save(name, "sklearn", f"{name}.pkl", model.predict, data, train_time)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    main(ap.parse_args().models)
