# Results summary

## Model comparison (test set, sorted by Macro F1)

| Model | Type | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Train time (s) | Inference (ms/1000 rows) |
|---|---|---|---|---|---|---|---|---|
| XGBoost | sklearn | 0.9997 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 59.57 | 23.89 |
| HistGradientBoosting | sklearn | 0.9995 | 0.9995 | 0.9995 | 0.9995 | 0.9996 | 61.68 | 40.97 |
| RandomForest | sklearn | 0.9989 | 0.999 | 0.9989 | 0.999 | 0.999 | 34.14 | 14.88 |
| MLP_sklearn | sklearn | 0.9989 | 0.9988 | 0.9989 | 0.9986 | 0.9989 | 203.26 | 9.27 |

**Best model:** XGBoost (Macro F1 = 0.9997, Accuracy = 0.9997)

## Per-class report: XGBoost

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 1.000 | 1.000 | 1.000 | 4500 |
| BruteForce | 0.999 | 1.000 | 1.000 | 1373 |
| DDoS | 1.000 | 1.000 | 1.000 | 4500 |
| DoS | 1.000 | 1.000 | 1.000 | 4500 |
| PortScan | 1.000 | 1.000 | 1.000 | 4500 |
