#!/usr/bin/env python
import os
import json
import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from src import config
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================

@dataclass
class Config:
    data_path: str = config.DATASET_PATH   # <-- change to your CSV path
    output_model_path: str = config.MLP_MODEL_PATH

    # Columns in your CSV
    feature_cols: tuple = (
        "cycle_idx",
        "q_NS_ema",
        "q_EW_ema",
        "ns_delay_prev",
        "ew_delay_prev",
        "prev_split",
        "prev_cycle",
    )
    target_cols: tuple = ("s_star", "C_star")

    # Train / val / test split (by proportion)
    train_ratio: float = 0.7
    val_ratio: float = 0.15   # rest is test

    # Training hyperparameters
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # MLP architecture
    hidden_sizes: tuple = (64, 64)
    dropout: float = 0.1

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4

    # Random seed
    seed: int = config.SEED

cfg = Config()


# =========================
# Reproducibility
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


# =========================
# Dataset
# =========================

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        """
        X: np.ndarray of shape (N, D)
        y: np.ndarray of shape (N, 2)  [s_star, C_star]
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Model
# =========================

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes=(64, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# Early Stopping
# =========================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# =========================
# Metrics
# =========================

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


# =========================
# Data Loading & Split
# =========================

def load_and_split_data(cfg: Config):
    df = pd.read_csv(cfg.data_path)

    # Ensure required columns exist
    for col in cfg.feature_cols + cfg.target_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    X = df[list(cfg.feature_cols)].values.astype(np.float32)
    y = df[list(cfg.target_cols)].values.astype(np.float32)

    N = X.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    train_end = int(N * cfg.train_ratio)
    val_end = int(N * (cfg.train_ratio + cfg.val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize inputs using train stats
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0  # avoid div by zero

    X_train_norm = (X_train - x_mean) / x_std
    X_val_norm = (X_val - x_mean) / x_std
    X_test_norm = (X_test - x_mean) / x_std

    # Optional: normalize outputs too (helps with mixed scales)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-8] = 1.0

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    train_ds = TrafficDataset(X_train_norm, y_train_norm)
    val_ds = TrafficDataset(X_val_norm, y_val_norm)
    test_ds = TrafficDataset(X_test_norm, y_test_norm)

    return train_ds, val_ds, test_ds, stats


# =========================
# Training Loop
# =========================

def train_model(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, stats = load_and_split_data(cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    in_dim = len(cfg.feature_cols)
    out_dim = len(cfg.target_cols)

    model = MLPRegressor(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, cfg.num_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        # Early stopping step
        early_stopper.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # =========================
    # Final Evaluation on Test
    # =========================
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    if len(all_preds) == 0:
        print("Warning: Test set is empty. Skipping test evaluation.")
        test_metrics = {}
    else:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # De-normalize outputs before computing interpretable metrics
        y_mean = torch.from_numpy(stats["y_mean"]).float()
        y_std = torch.from_numpy(stats["y_std"]).float()
        preds_denorm = all_preds * y_std + y_mean
        targets_denorm = all_targets * y_std + y_mean

        # Split into s_star and C_star components
        pred_s = preds_denorm[:, 0]
        pred_C = preds_denorm[:, 1]
        target_s = targets_denorm[:, 0]
        target_C = targets_denorm[:, 1]

        mae_s = mae(pred_s, target_s)
        mae_C = mae(pred_C, target_C)
        rmse_s = rmse(pred_s, target_s)
        rmse_C = rmse(pred_C, target_C)

        test_metrics = {
            "test_mae_s_star": mae_s,
            "test_mae_C_star": mae_C,
            "test_rmse_s_star": rmse_s,
            "test_rmse_C_star": rmse_C,
            "num_test_samples": int(target_s.shape[0]),
        }

        print("\n=== Test Metrics (denormalized) ===")
        print(f"MAE s_star: {mae_s:.6f}")
        print(f"RMSE s_star: {rmse_s:.6f}")
        print(f"MAE C_star: {mae_C:.6f}")
        print(f"RMSE C_star: {rmse_C:.6f}")
        print(f"Test samples: {test_metrics['num_test_samples']}")

    # =========================
    # Save best model + stats
    # =========================
    save_payload = {
        "model_state_dict": model.state_dict(),
        "input_mean": stats["x_mean"],
        "input_std": stats["x_std"],
        "output_mean": stats["y_mean"],
        "output_std": stats["y_std"],
        "config": asdict(cfg),
        "test_metrics": test_metrics,
    }

    torch.save(save_payload, cfg.output_model_path)
    print(f"\nSaved best model + stats to: {cfg.output_model_path}")


if __name__ == "__main__":
    train_model(cfg)
