"""Run all tests:   python -m unittest discover -s tests -v

Unit tests check the logic of single functions. The end-to-end test runs the whole
project (generate -> preprocess -> train -> evaluate -> predict -> dashboard smoke)
inside a temporary folder using small/fast settings, so it never touches your data."""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, predict, preprocess
from src.generate_sample_data import FEATURES
from src.utils import signed_log1p


def make_raw(n=60, seed=0):
    """Small raw-format dataframe: leading-space columns, IP columns, dirty values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(FEATURES))) * 100, columns=FEATURES)
    df["Label"] = ["BENIGN", "DDoS", "PortScan", "DoS Hulk", "FTP-Patator", "Bot"] * (n // 6)
    return df


class TestLabels(unittest.TestCase):
    def test_mapping(self):
        m = preprocess.map_label
        self.assertEqual(m("BENIGN"), "Benign")
        self.assertEqual(m(" DDoS "), "DDoS")
        self.assertEqual(m("PortScan"), "PortScan")
        for dos in ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"]:
            self.assertEqual(m(dos), "DoS")
        self.assertEqual(m("FTP-Patator"), "BruteForce")
        self.assertEqual(m("SSH-Patator"), "BruteForce")

    def test_ddos_is_not_dos(self):
        self.assertEqual(preprocess.map_label("DDoS"), "DDoS")

    def test_unmapped_labels_are_dropped(self):
        for x in ["Bot", "Infiltration", "Heartbleed", "Web Attack \x96 XSS", "", None, float("nan")]:
            self.assertIsNone(preprocess.map_label(x), x)


class TestCleaning(unittest.TestCase):
    def test_clean_frame(self):
        df = make_raw(60)
        df["Flow ID"] = "x"
        df["Source IP"] = "1.2.3.4"
        df["Timestamp"] = "t"
        df.loc[0, "Flow Bytes/s"] = np.inf          # DDoS? row0 is BENIGN -> removed
        df.loc[7, "Flow Packets/s"] = np.nan
        df = pd.concat([df, df.iloc[[2, 3]]], ignore_index=True)   # 2 exact duplicates
        df.columns = [" " + c for c in df.columns]                  # leading spaces like real files
        out = preprocess.clean_frame(df)
        self.assertNotIn("Source IP", out.columns)
        self.assertNotIn("Flow ID", out.columns)
        self.assertNotIn("Timestamp", out.columns)
        self.assertIn("Destination Port", out.columns)              # names were stripped
        self.assertFalse(np.isinf(out.drop(columns="Label").to_numpy()).any())
        self.assertFalse(out.isna().any().any())
        self.assertFalse(out.duplicated().any())
        self.assertNotIn("Bot", set(out["Label"]))                  # unmapped dropped
        self.assertEqual(set(out["Label"]), {"Benign", "DDoS", "PortScan", "DoS", "BruteForce"})
        self.assertEqual(len(out), 50 - 2)                          # 10 Bot rows, 2 dirty rows removed

    def test_missing_label_column(self):
        with self.assertRaises(ValueError):
            preprocess.clean_frame(make_raw(12).drop(columns="Label"))


class TestSplitAndScale(unittest.TestCase):
    def setUp(self):
        raw = make_raw(600, seed=1)
        self.df = preprocess.clean_frame(raw)
        (self.arrays, self.le, self.scaler, self.feats, self.medians,
         self.X_te_raw, self.y_te) = preprocess.split_and_scale(self.df, verbose=False)

    def test_sizes_and_stratification(self):
        n = len(self.df)
        tr, va, te = (len(self.arrays[k]) for k in ("y_train", "y_val", "y_test"))
        self.assertEqual(tr + va + te, n)
        self.assertAlmostEqual(tr / n, 0.70, delta=0.02)
        self.assertAlmostEqual(va / n, 0.15, delta=0.02)
        full = pd.Series(self.le.inverse_transform(np.r_[self.arrays["y_train"], self.arrays["y_val"], self.arrays["y_test"]])).value_counts(normalize=True)
        part = pd.Series(self.le.inverse_transform(self.arrays["y_test"])).value_counts(normalize=True)
        for c in full.index:
            self.assertAlmostEqual(full[c], part[c], delta=0.03)

    def test_scaler_fit_on_train_only(self):
        Xtr = self.arrays["X_train"]
        np.testing.assert_allclose(Xtr.mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(Xtr.std(axis=0), 1, atol=1e-6)
        # test set is NOT exactly standardised -> it was not used to fit the scaler
        self.assertGreater(np.abs(self.arrays["X_test"].mean(axis=0)).max(), 1e-6)

    def test_no_row_overlap_between_splits(self):
        tr = {tuple(r) for r in np.round(self.arrays["X_train"], 9)}
        te = {tuple(r) for r in np.round(self.arrays["X_test"], 9)}
        va = {tuple(r) for r in np.round(self.arrays["X_val"], 9)}
        self.assertFalse(tr & te)
        self.assertFalse(tr & va)
        self.assertFalse(va & te)

    def test_label_not_in_features(self):
        self.assertNotIn("Label", self.feats)
        self.assertEqual(self.arrays["X_train"].shape[1], len(FEATURES))

    def test_tiny_classes_are_dropped(self):
        df = self.df.copy()
        extra = df[df["Label"] == "DDoS"].head(3).copy()
        extra["Label"] = "Tiny"
        out = preprocess.split_and_scale(pd.concat([df, extra], ignore_index=True), verbose=False)
        self.assertNotIn("Tiny", list(out[1].classes_))


class TestBuildMatrix(unittest.TestCase):
    def setUp(self):
        raw = preprocess.clean_frame(make_raw(600, seed=2))
        (arrays, le, scaler, feats, medians, X_te_raw, y_te) = preprocess.split_and_scale(raw, verbose=False)
        self.art = {"scaler": scaler, "class_names": list(le.classes_),
                    "feature_names": feats, "medians": medians}
        self.X_te_raw, self.X_test = X_te_raw, arrays["X_test"]
        self.feats = feats

    def frame(self):
        return pd.DataFrame(self.X_te_raw, columns=self.feats)

    def test_matches_training_transform(self):
        np.testing.assert_allclose(predict.build_matrix(self.frame(), self.art), self.X_test)

    def test_column_order_and_extra_columns_do_not_matter(self):
        df = self.frame()
        df["Label"] = "x"
        df["Source IP"] = "1.1.1.1"
        df = df[list(reversed(df.columns))]
        df.columns = [" " + c for c in df.columns]
        np.testing.assert_allclose(predict.build_matrix(df, self.art), self.X_test)

    def test_inf_and_nan_are_filled(self):
        df = self.frame()
        df = df.astype({self.feats[2]: object})   # allow a text value in that column
        df.iloc[0, 0] = np.inf
        df.iloc[1, 1] = np.nan
        df.iloc[2, 2] = "garbage"
        X = predict.build_matrix(df, self.art)
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(X.shape, self.X_test.shape)

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            predict.build_matrix(self.frame().drop(columns=[self.feats[0]]), self.art)

    def test_signed_log1p(self):
        np.testing.assert_allclose(signed_log1p([-1, 0, 1]), [-np.log(2), 0, np.log(2)])


class TestEndToEnd(unittest.TestCase):
    """Whole project in a temporary folder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.env = dict(os.environ, TC_ROOT=str(cls.root), TC_FAST="1", PYTHONPATH=str(ROOT))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def sh(self, *args):
        r = subprocess.run([sys.executable, *args], cwd=ROOT, env=self.env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f"{args} failed:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    def test_full_pipeline(self):
        self.sh("-m", "src.generate_sample_data", "--scale", "0.6", "--out", str(self.root / "data" / "raw"))
        self.assertEqual(len(list((self.root / "data" / "raw").glob("*.csv"))), 6)

        out = self.sh("-m", "src.preprocess")
        self.assertIn("Classes: ['Benign', 'BruteForce', 'DDoS', 'DoS', 'PortScan']", out)
        for f in ["data/processed/data.npz", "models/scaler.pkl", "models/label_encoder.pkl",
                  "models/feature_names.json", "models/medians.json", "data/sample/demo_test.csv"]:
            self.assertTrue((self.root / f).exists(), f)

        self.sh("-m", "src.train_ml")
        self.sh("-m", "src.train_dl", "--backend", "sklearn")
        self.sh("-m", "src.evaluate")

        for name in ["RandomForest", "HistGradientBoosting", "MLP_sklearn"]:
            m = json.loads((self.root / "results" / f"metrics_{name}.json").read_text())
            self.assertGreater(m["macro_f1"], 0.6, name)
            self.assertEqual(np.asarray(m["confusion_matrix"]).shape, (5, 5))
            self.assertEqual(int(np.sum(m["confusion_matrix"])), int(sum(v["support"] for v in m["per_class"].values())))
            self.assertTrue((self.root / "results" / "figures" / f"confusion_{name}.png").exists())
        for f in ["comparison.csv", "summary.md", "figures/model_comparison.png",
                  "figures/class_distribution.png", "figures/feature_importance.png"]:
            self.assertTrue((self.root / "results" / f).exists(), f)

        # CLI prediction on the demo file is consistent with the test metrics
        out_csv = self.root / "preds.csv"
        self.sh("-m", "src.predict", str(self.root / "data" / "sample" / "demo_test.csv"),
                "--model", "RandomForest", "--out", str(out_csv))
        d = pd.read_csv(out_csv)
        acc = (d["Label"] == d["Prediction"]).mean()
        self.assertGreater(acc, 0.6)

        # dashboard (fake streamlit) smoke test against this temp project
        out = self.sh("tests/run_dashboard_smoke.py")
        self.assertIn("ALL DASHBOARD SMOKE CHECKS PASSED", out)

    def test_run_without_data_gives_clear_error(self):
        empty = Path(self.tmp.name) / "empty_case"
        env = dict(self.env, TC_ROOT=str(empty))
        r = subprocess.run([sys.executable, "-m", "src.preprocess"], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("No CSV files", r.stderr)


if __name__ == "__main__":
    unittest.main()
