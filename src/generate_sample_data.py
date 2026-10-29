"""Generate a SYNTHETIC dataset in the CIC-IDS2017 (MachineLearningCVE) CSV format.

Why: lets you test the whole pipeline without downloading the 1+ GB real dataset.
The numbers are random (NOT real network traffic), so accuracy on this data only
proves the code works. For the report, use the real CIC-IDS2017 CSVs.

Usage:  python -m src.generate_sample_data [--scale 1.0] [--out data/raw]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, RANDOM_STATE

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
assert len(FEATURES) == 78, len(FEATURES)

# How far apart the classes are (smaller = harder). tuned so models reach roughly 90-98% accuracy.
SEPARATION = 0.45

# (raw label, group, base number of rows, destination port)
CLASSES = [
    ("BENIGN", "benign", 3000, None),
    ("DDoS", "ddos", 1500, 80),
    ("PortScan", "portscan", 1500, None),
    ("DoS Hulk", "dos", 900, 80),
    ("DoS slowloris", "dos", 400, 80),
    ("FTP-Patator", "brute", 400, 21),
    ("SSH-Patator", "brute", 300, 22),
    ("Bot", "bot", 200, None),                                  # unmapped -> must be dropped
    ("Web Attack " + chr(0x96) + " Brute Force", "web", 150, 80),  # non-UTF8 label, dropped
]

# which raw file each class goes to (mimics CIC-IDS2017 day files)
FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv": ["BENIGN"],
    "Tuesday-WorkingHours.pcap_ISCX.csv": ["FTP-Patator", "SSH-Patator"],
    "Wednesday-workingHours.pcap_ISCX.csv": ["DoS Hulk", "DoS slowloris"],
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": ["DDoS"],
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": ["PortScan"],
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": ["Bot", "Web Attack " + chr(0x96) + " Brute Force"],
}


def generate(out_dir=DATA_RAW, scale=1.0, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURES)
    base_mu = rng.uniform(1.0, 6.0, n_feat)
    base_sigma = rng.uniform(0.8, 1.6, n_feat)
    informative = rng.choice(n_feat, 35, replace=False)

    group_offset = {}
    for g in ["benign", "ddos", "portscan", "dos", "brute", "bot", "web"]:
        off = np.zeros(n_feat)
        off[informative] = rng.normal(0, SEPARATION, informative.size)
        group_offset[g] = off

    frames = {}
    for label, group, n_base, port in CLASSES:
        n = max(int(n_base * scale), 20)
        # variants inside a group differ slightly from each other
        off = group_offset[group] + rng.normal(0, 0.15, n_feat) * (label != "BENIGN")
        mu = base_mu + off
        X = np.exp(rng.normal(mu, base_sigma, size=(n, n_feat))) - 1.0
        X = np.round(np.clip(X, 0, None), 3)
        df = pd.DataFrame(X, columns=FEATURES)
        # destination port: mostly the class' typical port, sometimes random
        common = np.array([80, 443, 53, 8080, 22, 21, 25, 3389])
        if port is not None:
            ports = np.where(rng.random(n) < 0.8, port, rng.choice(common, n))
        elif group == "portscan":
            ports = rng.integers(1, 65535, n)
        else:
            ports = rng.choice(common, n)
        df["Destination Port"] = ports
        df["Label"] = label
        frames[label] = df

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, labels in FILES.items():
        df = pd.concat([frames[l] for l in labels], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # dirty the data like the real dataset: inf, NaN, duplicates
        k = max(int(0.01 * len(df)), 1)
        idx = rng.choice(len(df), k, replace=False)
        df.loc[idx, "Flow Bytes/s"] = np.inf
        idx = rng.choice(len(df), k, replace=False)
        df.loc[idx, "Flow Packets/s"] = np.nan
        dup = df.sample(frac=0.02, random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)
        # real files have a leading space in every column name
        df.columns = [" " + c for c in df.columns]
        path = out_dir / fname
        df.to_csv(path, index=False, encoding="latin-1")
        written.append(path)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=1.0, help="row multiplier (default 1.0)")
    ap.add_argument("--out", default=str(DATA_RAW), help="output folder")
    args = ap.parse_args()
    files = generate(args.out, args.scale)
    for f in files:
        print("wrote", f)
