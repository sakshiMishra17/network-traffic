"""Run the whole project with ONE command.

    python run_pipeline.py            # uses the CSV files you put in data/raw
    python run_pipeline.py --demo     # no dataset yet? create synthetic demo data first

Steps: (generate demo data) -> preprocess -> train_ml -> train_dl -> evaluate
Then start the dashboard:   streamlit run app/dashboard.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(title, *args):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\nStep failed: {title}")
    print(f"-- done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="generate synthetic demo data if data/raw is empty")
    ap.add_argument("--max-per-class", type=int, default=None, help="cap rows per class")
    ap.add_argument("--skip-dl", action="store_true", help="skip deep learning training")
    a = ap.parse_args()

    raw = ROOT / "data" / "raw"
    if a.demo and not list(raw.glob("*.csv")):
        step("1/5 Generate synthetic demo data", "-m", "src.generate_sample_data")
    pre = ["-m", "src.preprocess"] + (["--max-per-class", str(a.max_per_class)] if a.max_per_class else [])
    step("2/5 Preprocess", *pre)
    step("3/5 Train ML models", "-m", "src.train_ml")
    if not a.skip_dl:
        step("4/5 Train DL models", "-m", "src.train_dl")
    step("5/5 Evaluate and make graphs", "-m", "src.evaluate")
    print("\nAll done. Results: results/summary.md and results/figures/")
    print("Start the dashboard with:  streamlit run app/dashboard.py")
