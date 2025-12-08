#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src import config

# ========= Config =========
DATA_PATH = config.DATASET_PATH
MODEL_PATH = config.MLP_MODEL_PATH

FEATURE_COLS = [
    "cycle_idx", "q_NS_ema", "q_EW_ema",
    "ns_delay_prev", "ew_delay_prev",
    "prev_split", "prev_cycle",
]
TARGET_COLS = ["s_star", "C_star"]

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 200
PATIENCE = 20
HIDDEN = [64, 64]
DROPOUT = 0.1
SEED = config.SEED


# ========= Utils =========
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden, dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ========= Data =========
def load_dataset(path: str, feature_cols, target_cols):
    df = pd.read_csv(path)

    for col in feature_cols + target_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in CSV")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    return X, y


def split_data(X, y, seed: int):
    # 70% train, 15% val, 15% test via 2-step split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_n = x_scaler.fit_transform(X_train)
    X_val_n = x_scaler.transform(X_val)
    X_test_n = x_scaler.transform(X_test)

    y_train_n = y_scaler.fit_transform(y_train)
    y_val_n = y_scaler.transform(y_val)
    y_test_n = y_scaler.transform(y_test)

    return (
        (X_train_n, y_train_n),
        (X_val_n, y_val_n),
        (X_test_n, y_test_n),
        x_scaler,
        y_scaler,
    )


def make_dataloaders(X_train_n, y_train_n, X_val_n, y_val_n, X_test_n, y_test_n, batch_size):
    train_ds = TensorDataset(torch.tensor(X_train_n), torch.tensor(y_train_n))
    val_ds   = TensorDataset(torch.tensor(X_val_n),   torch.tensor(y_val_n))
    test_ds  = TensorDataset(torch.tensor(X_test_n),  torch.tensor(y_test_n))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ========= Train / Eval =========
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    lr: float,
    epochs: int,
    patience: int,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch:03d}  Val Loss = {val_loss:.6f}")

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def evaluate_model(model, test_loader, y_scaler, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    if not preds:
        print("No test samples, skipping metrics.")
        return {}

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    # de-normalize
    preds_den = y_scaler.inverse_transform(preds)
    trues_den = y_scaler.inverse_transform(trues)

    mae_s = mean_absolute_error(trues_den[:, 0], preds_den[:, 0])
    mae_C = mean_absolute_error(trues_den[:, 1], preds_den[:, 1])
    rmse_s = root_mean_squared_error(trues_den[:, 0], preds_den[:, 0])
    rmse_C = root_mean_squared_error(trues_den[:, 1], preds_den[:, 1])

    metrics = {
        "mae_s_star": mae_s,
        "mae_C_star": mae_C,
        "rmse_s_star": rmse_s,
        "rmse_C_star": rmse_C,
        "num_test": int(trues_den.shape[0]),
    }

    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    return metrics


def save_artifacts(model, x_scaler, y_scaler, metrics, model_path: str):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "test_metrics": metrics,
        },
        model_path,
    )
    print(f"\nSaved model to {model_path}")


# ========= Main =========
def train_de_ml():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) load
    X, y = load_dataset(DATA_PATH, FEATURE_COLS, TARGET_COLS)

    # 2) split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, SEED)

    # 3) normalize
    (X_train_n, y_train_n), (X_val_n, y_val_n), (X_test_n, y_test_n), x_scaler, y_scaler = normalize_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # 4) dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train_n, y_train_n,
        X_val_n, y_val_n,
        X_test_n, y_test_n,
        BATCH_SIZE,
    )

    # 5) model
    model = MLP(
        in_dim=len(FEATURE_COLS),
        out_dim=len(TARGET_COLS),
        hidden=HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    # 6) train
    model, best_val = train_model(
        model,
        train_loader,
        val_loader,
        device,
        lr=LR,
        epochs=EPOCHS,
        patience=PATIENCE,
    )
    print(f"Best val loss: {best_val:.6f}")

    # 7) eval
    metrics = evaluate_model(model, test_loader, y_scaler, device)

    # 8) save
    save_artifacts(model, x_scaler, y_scaler, metrics, MODEL_PATH)
