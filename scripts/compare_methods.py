"""
Overlay ROME vs MEMIT degradation curves on the same axes, so you can
directly compare how robust each editing method is under sequential edits.

Run (after running both):
    python run_sequential_edit.py --method ROME
    python run_sequential_edit.py --method MEMIT
    python compare_methods.py --window 10
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATHS = {
    "ROME": "./results/sequential_edit_rome.json",
    "MEMIT": "./results/sequential_edit_memit.json",
}
OUTPUT_PATH = "./results/method_comparison.png"


def windowed_mean(values, window):
    values = np.array(values, dtype=float)
    n_windows = len(values) // window
    trimmed = values[: n_windows * window]
    return trimmed.reshape(n_windows, window).mean(axis=1)


def load_metric_series(path):
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rewrite_acc = [m["post"]["rewrite_acc"][0] for m in metrics]
    rephrase_acc = [m["post"]["rephrase_acc"][0] for m in metrics]
    locality_acc = [m["post"]["locality"]["neighborhood_acc"][0] for m in metrics]
    return rewrite_acc, rephrase_acc, locality_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()
    w = args.window

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    metric_names = ["Reliability (rewrite_acc)", "Generalization (rephrase_acc)", "Locality (neighborhood_acc)"]
    colors = {"ROME": "tab:blue", "MEMIT": "tab:red"}

    all_series = {}
    for method, path in RESULTS_PATHS.items():
        all_series[method] = load_metric_series(path)

    for metric_idx, ax in enumerate(axes):
        for method in RESULTS_PATHS:
            series = all_series[method][metric_idx]
            series_w = windowed_mean(series, w)
            x = [(i + 1) * w for i in range(len(series_w))]
            ax.plot(x, series_w, marker="o", label=method, color=colors[method])
        ax.set_title(metric_names[metric_idx])
        ax.set_xlabel("Cumulative number of edits")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(f"Accuracy (averaged over each {w}-edit window)")
    fig.suptitle("ROME vs MEMIT under sequential editing (Qwen2.5-7B)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print("Saved plot to", OUTPUT_PATH)


if __name__ == "__main__":
    main()