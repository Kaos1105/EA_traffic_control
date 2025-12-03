from src import config
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

def get_green_split(s, C):
    g_main = max(config.GREEN_MIN, round((C - config.LOST_TIME) * s))
    g_cross = max(config.GREEN_MIN, (C - config.LOST_TIME) - g_main)
    return g_main, g_cross

def log_cycle_result(
    cycle,
    s,
    C,
    O1,
    O2,
    score_best,
    elapsed_s,
    suffix="",
    out_dir=config.LOG_DIR,
    prefix="DE_cycle",
):
    """
    Append one cycle results to a CSV log file
    """
    # --- Create output folder
    os.makedirs(out_dir, exist_ok=True)

    # Main summary file (aggregated cycle-level results)
    summary_file = os.path.join(out_dir, f"{prefix}_summary_{suffix}.csv")
    write_header = not os.path.exists(summary_file)

    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "cycle", "s", "C", "O1", "O2",
                "score_best", "elapsed_s"
            ])
        writer.writerow([
            cycle, round(s, 3), round(C, 1), round(O1, 3), round(O2, 3),
            round(score_best, 4), round(elapsed_s, 2)
        ])

    return summary_file

def visualize_pow_results(log_dir, suffix="", prefix="traffic_DE"):
    """
    Visualize core PoW evidence from your DE summary CSV.
    Expects: CSV created by log_cycle_result()
    Plots:
        1. Cycle length evolution (adaptivity)
        2. Objective trends (O1, O2)
        3. Score trend (convergence)
        4. Optimization runtime per cycle
        5. Baseline vs. EA improvement summary (optional if baseline exists)
    """
    # --- Load summary CSV
    summary_file = os.path.join(log_dir, f"{prefix}_summary_{suffix}.csv")
    if not os.path.exists(summary_file):
        print(f"❌ Summary file not found: {summary_file}")
        return

    df = pd.read_csv(summary_file)
    print(f"Loaded {len(df)} cycles from {summary_file}")

    # --- 1. Cycle length evolution
    plt.figure(figsize=(7,4))
    plt.plot(df["cycle"], df["C"], marker='o')
    plt.xlabel("Cycle #")
    plt.ylabel("Cycle length (s)")
    plt.title("Adaptive cycle length over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 2. Objective trends
    plt.figure(figsize=(7,4))
    plt.plot(df["cycle"], df["O1"], label="Max Avg delay (O1)")
    plt.plot(df["cycle"], df["O2"], label="Fairness (O2)")
    plt.xlabel("Cycle #")
    plt.ylabel("Objective value")
    plt.title("Objective trends over cycles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 3. Score trend
    plt.figure(figsize=(7,4))
    plt.plot(df["cycle"], df["score_best"], label="Score (per cycle)", color='orange')
    plt.xlabel("Cycle #")
    plt.ylabel("Score")
    plt.title("Optimization score per cycle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 4. Runtime feasibility
    plt.figure(figsize=(7,4))
    plt.plot(df["cycle"], df["elapsed_s"], label="Runtime per optimization", color='red')
    plt.axhline(y=15, color='gray', linestyle='--', label="15s real-time bound")
    plt.xlabel("Cycle #")
    plt.ylabel("Computation time (s)")
    plt.title("Optimization runtime per cycle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 5. Summary statistics
    print("\n=== Summary statistics ===")
    print(df[["s", "C","O1","O2","score_best","elapsed_s"]].describe().round(3))
    avg_runtime = df["elapsed_s"].mean()
    print(f"\n✅  Average runtime per DE optimization: {avg_runtime:.2f}s")
    print("✅ Visualization complete — core PoW evidence generated.")

def compare_logs_by_time(log_dir=config.LOG_DIR, baseline_prefix='traffic_baseline', de_prefix ='traffic_DE', baseline_suffix = "", de_suffix = ""):
    """
    Compare DE and baseline log files by plotting s, C, O1, and O2 
    against time lapse (cycle × C).

    Parameters:
        de_path (str): Path to DE log CSV file
        baseline_path (str): Path to baseline log CSV file
    """

    de_path = os.path.join(log_dir, f"{de_prefix}_summary_{de_suffix}.csv")
    baseline_path = os.path.join(log_dir, f"{baseline_prefix}_summary_{baseline_suffix}.csv")
    if not os.path.exists(de_path):
        print(f"❌ Summary file not found: {de_path}")
        return
    if not os.path.exists(baseline_path):
        print(f"❌ Summary file not found: {baseline_path}")
        return
    # Load CSVs
    de = pd.read_csv(de_path)
    baseline = pd.read_csv(baseline_path)

     # Ensure same length / alignment
    min_len = min(len(de), len(baseline))
    de = de.head(min_len)
    baseline = baseline.head(min_len)

     # Variables to compare
    vars_to_plot = ["s", "C", "O1", "O2", "score_best"]

    # Plot each variable
    for v in vars_to_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(de["cycle"], de[v], label="DE", linewidth=2)
        plt.plot(baseline["cycle"], baseline[v], label="Baseline", linestyle="--", linewidth=2)
        plt.xlabel("Cycle")
        plt.ylabel(v)
        plt.title(f"Comparison of {v} over Cycle")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def compare_score_best(log_dir=config.LOG_DIR, baseline_prefix='traffic_baseline', de_prefix ='traffic_DE', baseline_suffix = "", de_suffix = ""):
    """
    Compare DE vs Baseline based on score_best over cycles.
    Shows a line plot and prints basic improvement summary.
    
    Parameters:
        de_path (str): Path to DE log CSV file
        baseline_path (str): Path to baseline log CSV file
    """
    de_path = os.path.join(log_dir, f"{de_prefix}_summary_{de_suffix}.csv")
    baseline_path = os.path.join(log_dir, f"{baseline_prefix}_summary_{baseline_suffix}.csv")
    if not os.path.exists(de_path):
        print(f"❌ Summary file not found: {de_path}")
        return
    if not os.path.exists(baseline_path):
        print(f"❌ Summary file not found: {baseline_path}")
        return
    # Load CSVs
    de = pd.read_csv(de_path)
    baseline = pd.read_csv(baseline_path)

    # Load the CSVs
    de = pd.read_csv(de_path)
    baseline = pd.read_csv(baseline_path)

    # Ensure same length / alignment
    min_len = min(len(de), len(baseline))
    de = de.head(min_len)
    baseline = baseline.head(min_len)

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.plot(de["cycle"], de["score_best"], label="DE", linewidth=2)
    plt.plot(baseline["cycle"], baseline["score_best"], label="Baseline", linestyle="--", linewidth=2)
    plt.xlabel("Cycle")
    plt.ylabel("score_best")
    plt.title("Comparison of score_best: DE vs Baseline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print summary stats
    mean_de = de["score_best"].mean()
    mean_base = baseline["score_best"].mean()
    diff = mean_de - mean_base

    print(f"Average DE score_best: {mean_de:.4f}")
    print(f"Average Baseline score_best: {mean_base:.4f}")
    print(f"Δ Improvement: {diff:+.4f} ({(diff / mean_base) * 100:.2f}% change)")

    if diff < 0:
        print("✅ DE improves over Baseline.")
    elif diff > 0:
        print("❌ DE performs worse than Baseline.")
    else:
        print("⭐ DE and Baseline perform equally.")