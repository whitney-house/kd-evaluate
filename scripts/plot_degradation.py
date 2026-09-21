"""
Plot the degradation curve from a sequential-edit results file.

With enough cases (100+), plotting every single 0/1 point is too noisy to
read a trend from. Instead, this bins edits into consecutive windows
(default: 10 edits per window) and plots the WINDOW AVERAGE — this is what
actually reveals whether locality/generalization degrade as the number of
accumulated edits grows.

Run (after run_sequential_edit.py has produced results/sequential_edit_result.json):
    python plot_degradation.py --window 10
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "./results/sequential_edit_result.json"
OUTPUT_PATH = "./results/degradation_curve.png"


def windowed_mean(values, window):
    values = np.array(values, dtype=float)
    n_windows = len(values) // window
    trimmed = values[: n_windows * window]
    return trimmed.reshape(n_windows, window).mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=10, help="edits per averaging window")
    args = parser.parse_args()

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rewrite_acc = [m["post"]["rewrite_acc"][0] for m in metrics]
    rephrase_acc = [m["post"]["rephrase_acc"][0] for m in metrics]
    locality_acc = [m["post"]["locality"]["neighborhood_acc"][0] for m in metrics]

    w = args.window
    rewrite_w = windowed_mean(rewrite_acc, w)
    rephrase_w = windowed_mean(rephrase_acc, w)
    locality_w = windowed_mean(locality_acc, w)
    x = [(i + 1) * w for i in range(len(rewrite_w))]  # cumulative edit count at each window's end

    plt.figure(figsize=(9, 5))
    plt.plot(x, rewrite_w, marker="o", label="Reliability (rewrite_acc)")
    plt.plot(x, rephrase_w, marker="s", label="Generalization (rephrase_acc)")
    plt.plot(x, locality_w, marker="^", label="Locality (neighborhood_acc)")
    plt.xlabel("Cumulative number of edits")
    plt.ylabel(f"Accuracy (averaged over each {w}-edit window)")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Editing metrics vs. sequential edits (ROME, Qwen2.5-7B, window={w})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print("Saved plot to", OUTPUT_PATH)


if __name__ == "__main__":
    main()