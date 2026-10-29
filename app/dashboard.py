"""Streamlit dashboard: upload network-flow CSV -> classify each flow -> charts.

Run from the project folder:   streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config
from src.predict import (list_models, load_artifacts, load_model, predict_dataframe,
                         read_csv_flexible)
from src.preprocess import map_label

st.set_page_config(page_title="Network Traffic Classifier", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_model(name):
    return load_model(name)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def pie_chart(counts):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Traffic share by predicted class")
    fig.tight_layout()
    return fig


def bar_chart(counts):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("Flows")
    ax.set_title("Flows per predicted class")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def confusion_chart(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (uploaded file)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- header
st.title("🛡️ Network Traffic Classification")
st.caption("Machine Learning and Deep Learning on CIC-IDS2017 flow features: "
           "Benign, DDoS, DoS, PortScan, BruteForce")

try:
    art = get_artifacts()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

models = list_models()
if not models:
    st.error("No trained models found. Run `python -m src.train_ml` first.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
model_names = [m["name"] for m in models]
info = {m["name"]: m for m in models}
model_name = st.sidebar.selectbox(
    "Model", model_names,
    format_func=lambda n: f"{n}  (macro-F1 {info[n]['macro_f1']:.3f})")
source = st.sidebar.radio("Data source", ["Demo file", "Upload CSV"])
uploaded = None
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CIC-IDS2017 style CSV", type=["csv"])

tab_pred, tab_cmp, tab_about = st.tabs(["Predict", "Model comparison", "About"])

# ---------------------------------------------------------------- predict
with tab_pred:
    raw = None
    if source == "Demo file":
        demo = config.DATA_SAMPLE / "demo_test.csv"
        if demo.exists():
            raw = read_csv_flexible(demo)
            st.info(f"Using demo file ({len(raw)} flows, includes true labels).")
        else:
            st.warning("Demo file not found. Run `python -m src.preprocess` first.")
    elif uploaded is not None:
        try:
            raw = read_csv_flexible(uploaded)
        except Exception as e:  # unreadable / not a csv
            st.error(f"Could not read the file: {e}")
    else:
        st.info("Upload a CSV file from the sidebar.")

    if raw is not None:
        try:
            with st.spinner("Classifying flows..."):
                pred = predict_dataframe(raw, get_model(model_name), art)
        except (ValueError, FileNotFoundError, ImportError, OSError) as e:
            st.error(f"{e}")
            pred = None

        if pred is not None:
            counts = pred["Prediction"].value_counts()
            total = len(pred)
            attacks = int((pred["Prediction"] != "Benign").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total flows", f"{total}")
            c2.metric("Benign", f"{total - attacks}")
            c3.metric("Attacks detected", f"{attacks}")
            c4.metric("Avg confidence", f"{pred['Confidence'].mean():.1%}")

            left, right = st.columns(2)
            with left:
                show_fig(pie_chart(counts))
            with right:
                show_fig(bar_chart(counts))

            # If the file has a Label column, compare with the truth
            label_col = next((c for c in raw.columns if str(c).strip() == config.LABEL_COL), None)
            if label_col is not None:
                truth = raw[label_col].map(map_label)
                mask = truth.notna() & truth.isin(art["class_names"])
                if int(mask.sum()) > 0:
                    y_true = truth[mask].to_numpy()
                    y_pred = pred.loc[mask, "Prediction"].to_numpy()
                    st.subheader("Accuracy on this file (labels found)")
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                    m2.metric("Macro F1", f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
                    cl, cr = st.columns(2)
                    with cl:
                        show_fig(confusion_chart(y_true, y_pred, art["class_names"]))
                    with cr:
                        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(rep).T.round(3))

            st.subheader("Predictions")
            only_attacks = st.checkbox("Show only attacks", value=False)
            result = pd.concat([pred.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
            if only_attacks:
                result = result[result["Prediction"] != "Benign"]
            st.dataframe(result.head(500))
            st.caption(f"Showing up to 500 of {len(result)} rows. Download has all rows.")
            st.download_button("Download predictions (CSV)",
                               data=result.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")

# ------------------------------------------------------------- comparison
with tab_cmp:
    comp = config.RESULTS_DIR / "comparison.csv"
    if comp.exists():
        st.subheader("Test-set results")
        st.dataframe(pd.read_csv(comp))
        for fname, title in [("model_comparison.png", "Accuracy and macro F1"),
                             ("per_class_f1.png", "Per-class F1"),
                             ("timing.png", "Training and inference time"),
                             ("class_distribution.png", "Class distribution"),
                             ("feature_importance.png", "Feature importance"),
                             (f"confusion_{model_name}.png", f"Confusion matrix: {model_name}")]:
            p = config.FIGURES_DIR / fname
            if p.exists():
                st.subheader(title)
                st.image(str(p))
    else:
        st.info("Run `python -m src.evaluate` to generate the comparison.")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        "**How it works**\n\n"
        "1. Network packets are grouped into flows; CICFlowMeter computes ~78 numeric features per flow.\n"
        "2. The CSV is cleaned (inf/NaN handled), log-scaled and standardised with the *training* scaler.\n"
        "3. The selected model predicts the traffic class for every flow, with a confidence score.\n\n"
        "**Classes:** Benign, DDoS, DoS, PortScan, BruteForce")
