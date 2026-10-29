"""Step 3: train deep learning models and save metrics.

* With PyTorch installed  -> MLP and 1D-CNN (class-weighted loss, best epoch by val macro-F1)
* Without PyTorch         -> falls back to scikit-learn MLPClassifier ("MLP_sklearn")

Usage:  python -m src.train_dl [--backend auto|torch|sklearn] [--epochs 20]
"""
import argparse
import time

import joblib
import numpy as np

from src import config
from src.config import RANDOM_STATE
from src.utils import evaluate_and_save, load_processed


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception as e:  # ImportError, or OSError on Windows when a torch DLL is missing
        print(f"PyTorch could not be loaded ({type(e).__name__}: {str(e)[:150]})")
        return False


# ------------------------------------------------------------- sklearn
def run_sklearn(data, epochs):
    from sklearn.neural_network import MLPClassifier
    print("\n=== Training MLP_sklearn (scikit-learn neural network) ===")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128), batch_size=256, max_iter=epochs,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
        random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    model.fit(data["X_train"], data["y_train"])
    train_time = time.perf_counter() - t0
    joblib.dump(model, config.MODELS_DIR / "MLP_sklearn.pkl")
    evaluate_and_save("MLP_sklearn", "sklearn", "MLP_sklearn.pkl", model.predict, data,
                      train_time, extra={"epochs": int(model.n_iter_)})


# --------------------------------------------------------------- torch
def run_torch(data, epochs):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, TensorDataset

    from src.dl_models import TorchPredictor, build_model

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device}")

    n_in = data["X_train"].shape[1]
    n_out = len(data["class_names"])
    batch = 512

    Xtr = torch.from_numpy(data["X_train"].astype(np.float32))
    ytr = torch.from_numpy(data["y_train"].astype(np.int64))
    # drop_last avoids a final batch of size 1, which crashes BatchNorm in train mode
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True,
                        drop_last=len(ytr) > batch)

    # class weights fight the class imbalance
    counts = np.bincount(data["y_train"], minlength=n_out).astype(np.float64)
    weights = counts.sum() / (n_out * np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    def predict_labels(model, X):
        model.eval()
        preds = []
        with torch.no_grad():
            Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            for i in range(0, len(Xt), 4096):
                logits = model(Xt[i:i + 4096].to(device))
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    for arch in ("MLP", "CNN1D"):
        print(f"\n=== Training {arch} ===")
        model = build_model(arch, n_in, n_out).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_f1, best_state = -1.0, None
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(yb)
            val_f1 = f1_score(data["y_val"], predict_labels(model, data["X_val"]),
                              average="macro", zero_division=0)
            print(f"  epoch {ep:2d}/{epochs}  loss={total / len(ytr):.4f}  val macro-F1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_time = time.perf_counter() - t0

        path = config.MODELS_DIR / f"{arch}.pt"
        torch.save({"arch": arch, "n_in": n_in, "n_out": n_out, "state_dict": best_state}, path)
        predictor = TorchPredictor(path)  # reload exactly as the dashboard will
        evaluate_and_save(arch, "torch", f"{arch}.pt", predictor.predict, data, train_time,
                          extra={"epochs": epochs, "best_val_macro_f1": float(best_f1)})


def main(backend="auto", epochs=None):
    config.ensure_dirs()
    data = load_processed()
    if backend == "auto":
        backend = "torch" if torch_available() else "sklearn"
        if backend == "sklearn":
            print("-> Using the scikit-learn MLP fallback instead of PyTorch.")
    if backend == "torch" and not torch_available():
        raise SystemExit("PyTorch is not working. Fix the install or use --backend sklearn.")
    if epochs is None:
        epochs = 3 if config.FAST else (60 if backend == "sklearn" else 20)
    print(f"DL backend: {backend}, epochs: {epochs}")
    (run_torch if backend == "torch" else run_sklearn)(data, epochs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    main(a.backend, a.epochs)
