import os
import time
from datetime import datetime

import numpy as np
from src.train_de_mlp import MLP
import torch
from torch import nn

from src import config   # adjust if needed


# ========= Match training config =========
HIDDEN = [64, 64]
DROPOUT = 0.1


# ========= Edge-like CPU constraints =========
def set_cpu_threads(n_threads: int):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)


def pctl(values_ms, p: float) -> float:
    return float(np.percentile(np.asarray(values_ms), p))


# ========= Load controller checkpoint =========
def load_mlp_controller_ckpt(model_path):
    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device)

    feature_cols = ckpt["feature_cols"]
    target_cols = ckpt["target_cols"]

    model = MLP(
        in_dim=len(feature_cols),
        out_dim=len(target_cols),
        hidden=HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, ckpt["x_scaler"], ckpt["y_scaler"], feature_cols, target_cols


# ========= ONNX =========
def export_to_onnx(model, in_dim, onnx_path):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    dummy = torch.randn(1, in_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
    )


def build_onnx_runner(onnx_path, threads):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    def run(x):
        return sess.run(None, {input_name: x})[0]

    return run


# ========= End-to-end prediction (edge) =========
def mlp_predict_plan_onnx(run, x_scaler, y_scaler, x_raw):
    x_norm = x_scaler.transform(x_raw).astype(np.float32)
    y_norm = run(x_norm)
    y_den = y_scaler.inverse_transform(y_norm)[0]

    s = float(np.clip(
        y_den[0],
        config.MIN_GREEN_SPLIT,
        1 - config.MIN_GREEN_SPLIT
    ))
    C = float(np.clip(
        y_den[1],
        config.MIN_CYCLE_LENGTH,
        config.MAX_CYCLE_LENGTH
    ))

    return s, C


# ========= Main benchmark =========
def seed_edge_benchmark(
    ckpt_path=config.MLP_MODEL_PATH,
    feature_csv=None,          # optional CSV with feature columns
    rows=256,
    warmup_iters=200,
    timed_iters=2000,
    threads=1,
    budget_ms=100.0,
    onnx_path=config.MLP_ONNX_MODEL_PATH,
):
    set_cpu_threads(threads)

    print("\n=== Edge Benchmark (ONNX, End-to-End) ===")
    print(f"Threads={threads}, Warmup={warmup_iters}, Iters={timed_iters}")

    # 1) Load checkpoint
    model, x_scaler, y_scaler, feature_cols, target_cols = load_mlp_controller_ckpt(
        ckpt_path
    )
    in_dim = len(feature_cols)

    print("Features:", feature_cols)
    print("Targets :", target_cols)

    # 2) Export ONNX
    export_to_onnx(model, in_dim, onnx_path)
    run_onnx = build_onnx_runner(onnx_path, threads)

    # 3) Inputs
    if feature_csv is not None:
        import pandas as pd
        df = pd.read_csv(feature_csv)
        X_raw = df[feature_cols].values.astype(np.float32)[:rows]
    else:
        X_raw = np.random.randn(rows, in_dim).astype(np.float32)

    # 4) Warmup
    for i in range(warmup_iters):
        xi = X_raw[i % rows : i % rows + 1]
        mlp_predict_plan_onnx(run_onnx, x_scaler, y_scaler, xi)

    # 5) Timed
    times = []
    t0_all = time.perf_counter()
    for i in range(timed_iters):
        xi = X_raw[i % rows : i % rows + 1]
        t0 = time.perf_counter()
        mlp_predict_plan_onnx(run_onnx, x_scaler, y_scaler, xi)
        times.append((time.perf_counter() - t0) * 1000)
    total = time.perf_counter() - t0_all

    # 6) Report
    print("\n==== Latency Report ====")
    print(f"Mean  : {np.mean(times):.3f} ms")
    print(f"p50   : {pctl(times, 50):.3f} ms")
    print(f"p95   : {pctl(times, 95):.3f} ms")
    print(f"p99   : {pctl(times, 99):.3f} ms")
    print(f"Max   : {np.max(times):.3f} ms")
    print(f"IPS   : {timed_iters / total:.1f} decisions/sec")
    print(f"Budget: {budget_ms:.1f} ms")
    print(f"PASS  : {pctl(times, 95) <= budget_ms}")
    print("========================\n")

    return times
