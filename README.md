# Network Traffic Classification using ML and DL

Classifies network flows as **Benign, DDoS, DoS, PortScan or BruteForce** from the 78 flow
features of the CIC-IDS2017 dataset. Compares classic ML (Random Forest, Gradient Boosting,
XGBoost) with deep learning (MLP, 1D-CNN) and ships a Streamlit dashboard.

## 1. Setup
```
python -m venv venv
venv\Scripts\activate          # Windows      (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```
`xgboost` and `torch` are optional. Without them the project still runs
(scikit-learn Gradient Boosting and a scikit-learn MLP are used instead).
Python 3.10+ recommended.

## 2. Get the dataset
Download **CIC-IDS2017 "MachineLearningCVE"** (8 CSV files, ~1 GB) from the UNB page or Kaggle
and copy the CSV files into `data/raw/`.
No dataset yet? Create synthetic demo data (same format, random numbers, only for testing the code):
```
python -m src.generate_sample_data
```

## 3. Run everything
```
python run_pipeline.py            # or: python run_pipeline.py --demo
streamlit run app/dashboard.py
```
Or step by step:

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m src.preprocess` | cleaning, label mapping, 70/15/15 split, scaling |
| 2 | `python -m src.train_ml` | Random Forest, Gradient Boosting, XGBoost |
| 3 | `python -m src.train_dl` | MLP + 1D-CNN (PyTorch) or sklearn MLP fallback |
| 4 | `python -m src.evaluate` | comparison table, confusion matrices, graphs |
| 5 | `streamlit run app/dashboard.py` | demo: upload CSV, see predictions |
| - | `python -m src.predict file.csv` | predictions from the command line |

Results: `results/summary.md`, `results/comparison.csv`, `results/figures/*.png`.

## 4. How it works
1. **Cleaning**: strip column names, drop identifier columns (IPs, Flow ID, Timestamp, Source Port)
   to avoid data leakage, remove `inf`/`NaN` rows and duplicates.
2. **Labels**: raw labels are mapped to 5 classes (all `DoS ...` variants -> DoS, FTP/SSH-Patator
   -> BruteForce). Bot, Infiltration, Web Attack and Heartbleed are dropped (too few rows).
   Edit `src/config.py` to change this. At most 30 000 rows per class are kept.
3. **Split and scale**: stratified 70/15/15; signed `log1p` + `StandardScaler` fitted on the
   **training set only**.
4. **Imbalance**: class weights (RF, GB, XGBoost via sample weights, PyTorch loss weights).
5. **Evaluation**: accuracy, **macro F1**, per-class precision/recall, confusion matrix,
   training time and inference time.
6. **Dashboard**: any CIC-IDS2017 style CSV is converted with exactly the same recipe; bad
   values are replaced by training medians so every flow gets a prediction.

## 5. Tests
```
python -m unittest discover -s tests -v
```
17 tests: label mapping, cleaning, no-leakage checks, inference == training transform,
and a full end-to-end run in a temporary folder (including a smoke test of the dashboard
with a fake Streamlit).

## 6. Project structure
```
src/config.py                 paths, class map, limits
src/generate_sample_data.py   synthetic CIC-style data for testing
src/preprocess.py             cleaning + split + scaling
src/train_ml.py, train_dl.py  model training
src/dl_models.py              PyTorch MLP / CNN1D
src/evaluate.py               tables and graphs
src/predict.py                inference helpers + CLI
app/dashboard.py              Streamlit app
tests/                        unit + end-to-end tests
```

## 7. Notes for the report / viva
* Results on the synthetic demo data are NOT results of the paper: use the real CSVs for the report.
* Why drop IPs/ports? The model would memorise addresses instead of traffic behaviour.
* Why macro F1? The classes are imbalanced; accuracy hides weak classes.
* Why fit the scaler on train only? Otherwise test information leaks into training.
* Limitations: flow features only (no raw packets), offline classification, 5 classes.
  Future work: live capture with CICFlowMeter, LSTM, more attack types.
