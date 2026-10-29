"""Runs app/dashboard.py against the fake streamlit in several scenarios.
Run with the project root as cwd:  python tests/run_dashboard_smoke.py"""
import io
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pandas as pd

from fake_streamlit import StopScript, make_fake
from src import config
from src.predict import list_models

DASH = str(ROOT / "app" / "dashboard.py")


def run(scenario):
    log = []
    sys.modules["streamlit"] = make_fake(scenario, log)
    try:
        runpy.run_path(DASH, run_name="__main__")
    except StopScript:
        pass
    return log


def errors(log):
    return [x[1] for x in log if isinstance(x, tuple)]


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        check.failed += 1
check.failed = 0


demo = pd.read_csv(config.DATA_SAMPLE / "demo_test.csv")

# 1. demo file with every available model
for m in list_models():
    log = run({"source": "Demo file", "model": m["name"]})
    check(not errors(log) and log.count("pyplot") >= 3 and "download_button" in log,
          f"demo file + {m['name']}: charts, accuracy section and download rendered")

# 2. upload the same data as bytes (like Streamlit's UploadedFile)
buf = io.BytesIO(demo.to_csv(index=False).encode())
log = run({"source": "Upload CSV", "uploaded": buf})
check(not errors(log) and "download_button" in log, "uploaded CSV works")

# 3. upload without Label column -> no accuracy section but predictions work
buf = io.BytesIO(demo.drop(columns=["Label"]).to_csv(index=False).encode())
log = run({"source": "Upload CSV", "uploaded": buf})
check(not errors(log) and "download_button" in log and log.count("pyplot") == 2,
      "CSV without Label: predictions only (2 charts)")

# 4. wrong CSV (missing columns) -> friendly error, no crash
buf = io.BytesIO(b"a,b,c\n1,2,3\n")
log = run({"source": "Upload CSV", "uploaded": buf})
check(any("missing" in e for e in errors(log)) and "download_button" not in log,
      "CSV with wrong columns -> friendly error")

# 5. garbage / empty file -> friendly error, no crash
log = run({"source": "Upload CSV", "uploaded": io.BytesIO(b"")})
check(len(errors(log)) == 1, "empty file -> friendly error")

# 6. nothing uploaded yet
log = run({"source": "Upload CSV", "uploaded": None})
check(not errors(log) and "info" in log and "download_button" not in log, "no upload -> info message")

# 7. CSV with only NaN/inf values still gets predictions
bad = demo.drop(columns=["Label"]).copy()
bad.iloc[:, :5] = float("inf")
buf = io.BytesIO(bad.to_csv(index=False).encode())
log = run({"source": "Upload CSV", "uploaded": buf})
check(not errors(log) and "download_button" in log, "inf values in upload are handled")

# 8. attack-only filter checkbox
log = run({"source": "Demo file", "checkbox": True})
check(not errors(log) and "download_button" in log, "'Show only attacks' filter works")

print("\nFAILED:" if check.failed else "\nALL DASHBOARD SMOKE CHECKS PASSED", check.failed or "")
sys.exit(1 if check.failed else 0)
