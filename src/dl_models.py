"""PyTorch model definitions + a loader used by evaluation and the dashboard.
Importing this module requires torch (only train_dl.py / predict.py import it, lazily)."""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected network: 78 -> 256 -> 128 -> n_classes."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Treats the feature vector as a 1-channel sequence and applies 1D convolutions."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveMaxPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_out),
        )

    def forward(self, x):                 # x: (batch, n_features)
        return self.fc(self.conv(x.unsqueeze(1)))


ARCHS = {"MLP": MLP, "CNN1D": CNN1D}


def build_model(arch, n_in, n_out):
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture {arch}. Choose from {list(ARCHS)}")
    return ARCHS[arch](n_in, n_out)


class TorchPredictor:
    """Loads a checkpoint saved by train_dl.py and exposes predict / predict_proba
    (same interface as sklearn models), always on CPU."""

    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.n_out = int(ckpt["n_out"])
        self.model = build_model(ckpt["arch"], int(ckpt["n_in"]), self.n_out)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict_proba(self, X, batch_size=4096):
        X = np.ascontiguousarray(X, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                logits = self.model(torch.from_numpy(X[i:i + batch_size]))
                outs.append(torch.softmax(logits, dim=1).numpy())
        return np.vstack(outs) if outs else np.zeros((0, self.n_out))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
