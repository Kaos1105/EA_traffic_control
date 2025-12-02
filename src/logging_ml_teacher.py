import csv
import os
import config

def init_dataset_file(path=config.DATASET_PATH):
    if not os.path.exists(path):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cycle_idx",
                "q_NS_ema",
                "q_EW_ema",
                "ns_delay_prev",
                "ew_delay_prev",
                "prev_split",
                "prev_cycle",
                "s_star",
                "C_star",
            ])


def append_sample(row, path=config.DATASET_PATH):
    with open(path, mode="a", newline="") as f:
        csv.writer(f).writerow(row)


def extract_temporal_features(
    q_ns_prev,
    q_ew_prev,
    q_ns_ema_prev,
    q_ew_ema_prev,
    ema_alpha=config.EMA_ALPHA,
):
    """
    Temporal features based on past cycle delays.

    q_ns_prev, q_ew_prev = ns_delay_prev, ew_delay_prev (cycle t-1)
    q_ns_ema_prev, q_ew_ema_prev = EMA of delays up to t-1
    """

    q_ns_ema = ema_alpha * q_ns_prev + (1.0 - ema_alpha) * q_ns_ema_prev
    q_ew_ema = ema_alpha * q_ew_prev + (1.0 - ema_alpha) * q_ew_ema_prev

    return {
        "q_NS_ema": q_ns_ema,
        "q_EW_ema": q_ew_ema,
        "q_NS_prev": q_ns_prev,
        "q_EW_prev": q_ew_prev,
    }

