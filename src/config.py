"""Central configuration for the Network Traffic Classification project."""
import os
from pathlib import Path

# Project root (can be overridden with env var TC_ROOT, used by the tests)
ROOT = Path(os.environ.get("TC_ROOT", Path(__file__).resolve().parents[1]))

DATA_RAW = ROOT / "data" / "raw"              # put CIC-IDS2017 CSV files here
DATA_PROCESSED = ROOT / "data" / "processed"  # train/val/test arrays
DATA_SAMPLE = ROOT / "data" / "sample"        # small demo CSV for the dashboard
MODELS_DIR = ROOT / "models"                  # saved models + scaler + encoder
RESULTS_DIR = ROOT / "results"                # metrics json, comparison csv
FIGURES_DIR = RESULTS_DIR / "figures"         # graphs (png)

RANDOM_STATE = 42
LABEL_COL = "Label"

# Columns that identify a flow but are not real traffic behaviour.
# Dropping them avoids data leakage (model memorising IPs / timestamps).
DROP_COLS = [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Timestamp", "Source Port", "Src Port", "SimillarHTTP", "Unnamed: 0",
]

# Mini-project classes. Raw CIC-IDS2017 labels are mapped to these 5 classes.
# Any label not covered here (Bot, Infiltration, Web Attack, Heartbleed) is dropped.
CLASS_MAP_EXACT = {
    "benign": "Benign",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
}
CLASS_MAP_PREFIX = {"dos": "DoS"}  # DoS Hulk, DoS GoldenEye, DoS slowloris, ...

# Max rows kept per class (per file and overall) so it trains fast on a laptop.
MAX_PER_CLASS = 30000
MIN_SAMPLES_PER_CLASS = 10

# TC_FAST=1 -> smaller/faster models (used by the tests)
FAST = os.environ.get("TC_FAST", "0") == "1"


def ensure_dirs():
    for d in (DATA_RAW, DATA_PROCESSED, DATA_SAMPLE, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
