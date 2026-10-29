"""Step 4: build comparison table and graphs from the saved metrics.

Usage:  python -m src.evaluate

Writes to results/ :
  comparison.csv, summary.md
  figures/class_distribution.png, model_comparison.png, timing.png,
  per_class_f1.png, confusion_<model>.png, feature_importance.png (if RandomForest exists)
"""
import glob

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.utils import load_json


def load_all_metrics():
    files = sorted(glob.glob(str(config.RESULTS_DIR / "metrics_*.json")))
    if not files:
        raise FileNotFoundError("No metrics found. Run train_ml / train_dl first.")
    metrics = [load_json(f) for f in files]
    return sorted(metrics, key=lambda m: -m["macro_f1"])


def plot_class_distribution(out):
    path = config.RESULTS_DIR / "class_counts.json"
    if not path.exists():
        return
    counts = load_json(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    names, vals = list(counts), list(counts.values())
    bars = ax.bar(names, vals, color="#4C72B0")
    ax.bar_label(bars, padding=2)
    ax.set_title("Class distribution (after cleaning)")
    ax.set_ylabel("Number of flows")
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_comparison(df, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    w = 0.38
    b1 = ax.bar(x - w / 2, df["Accuracy"], w, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + w / 2, df["Macro F1"], w, label="Macro F1", color="#DD8452")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylim(max(0.0, min(df["Macro F1"].min(), df["Accuracy"].min()) - 0.1), 1.03)
    ax.set_title("Model comparison (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_timing(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(df["Model"], df["Train time (s)"], color="#55A868")
    axes[0].set_title("Training time (s)")
    axes[1].bar(df["Model"], df["Inference (ms/1000 rows)"], color="#C44E52")
    axes[1].set_title("Inference time (ms per 1000 rows)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out / "timing.png", dpi=150)
    plt.close(fig)


def plot_confusion(m, out):
    cm = np.asarray(m["confusion_matrix"], dtype=float)
    names = m["class_names"]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{int(cm[i, j])}\n({norm[i, j]:.0%})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {m['model']}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / f"confusion_{m['model']}.png", dpi=150)
    plt.close(fig)


def plot_per_class_f1(metrics, out):
    names = metrics[0]["class_names"]
    grid = np.array([[m["per_class"][c]["f1-score"] for c in names] for m in metrics])
    fig, ax = plt.subplots(figsize=(7, 1.0 + 0.6 * len(metrics)))
    im = ax.imshow(grid, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m["model"] for m in metrics])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Per-class F1 score")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "per_class_f1.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(out, top=15):
    path = config.MODELS_DIR / "RandomForest.pkl"
    names_path = config.MODELS_DIR / "feature_names.json"
    if not path.exists() or not names_path.exists():
        return False
    model = joblib.load(path)
    names = np.asarray(load_json(names_path))
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names[order], imp[order], color="#8172B3")
    ax.set_title(f"Top {top} features (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)
    return True


def main():
    config.ensure_dirs()
    out = config.FIGURES_DIR
    metrics = load_all_metrics()

    df = pd.DataFrame([{
        "Model": m["model"],
        "Type": m["kind"],
        "Accuracy": round(m["accuracy"], 4),
        "Macro F1": round(m["macro_f1"], 4),
        "Weighted F1": round(m["weighted_f1"], 4),
        "Macro Precision": round(m["macro_precision"], 4),
        "Macro Recall": round(m["macro_recall"], 4),
        "Train time (s)": round(m["train_time_s"], 2),
        "Inference (ms/1000 rows)": round(m["infer_ms_per_1000"], 2),
    } for m in metrics])
    df.to_csv(config.RESULTS_DIR / "comparison.csv", index=False)

    plot_class_distribution(out)
    plot_comparison(df, out)
    plot_timing(df, out)
    plot_per_class_f1(metrics, out)
    for m in metrics:
        plot_confusion(m, out)
    has_fi = plot_feature_importance(out)

    best = metrics[0]
    lines = ["# Results summary", "", "## Model comparison (test set, sorted by Macro F1)", "",
             "| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    lines += ["", f"**Best model:** {best['model']} (Macro F1 = {best['macro_f1']:.4f}, "
                  f"Accuracy = {best['accuracy']:.4f})", "",
              f"## Per-class report: {best['model']}", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c in best["class_names"]:
        p = best["per_class"][c]
        lines.append(f"| {c} | {p['precision']:.3f} | {p['recall']:.3f} | "
                     f"{p['f1-score']:.3f} | {int(p['support'])} |")
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nBest model: {best['model']}")
    print(f"Figures saved in {out}" + ("" if has_fi else "  (no RandomForest -> no feature importance)"))


if __name__ == "__main__":
    main()
