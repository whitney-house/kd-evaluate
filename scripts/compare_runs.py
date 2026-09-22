"""
Overlay two sequential-edit runs on the same axes — works for comparing
methods (ROME vs MEMIT) or models (Qwen2.5-7B vs GPT2-XL), whichever two
result files you point it at.

Run:
    python compare_runs.py \
        --run results/sequential_edit_rome_qwen2.5-7b.json "Qwen2.5-7B" \
        --run results/sequential_edit_rome_gpt2-xl.json "GPT2-XL" \
        --window 10 \
        --title "ROME: Qwen2.5-7B vs GPT2-XL" \
        --output results/model_comparison.png
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


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
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("RESULTS_JSON", "LABEL"),
        required=True,
        help="pass twice, once per run to compare",
    )
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--title", default="Sequential editing comparison")
    parser.add_argument("--output", default="./results/comparison.png")
    args = parser.parse_args()
    w = args.window

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    metric_names = [
        "Reliability (rewrite_acc)",
        "Generalization (rephrase_acc)",
        "Locality (neighborhood_acc)",
    ]
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    runs = [(path, label) for path, label in args.run]

    for metric_idx, ax in enumerate(axes):
        for i, (path, label) in enumerate(runs):
            series = load_metric_series(path)[metric_idx]
            series_w = windowed_mean(series, w)
            x = [(j + 1) * w for j in range(len(series_w))]
            ax.plot(x, series_w, marker="o", label=label, color=colors[i % len(colors)])
        ax.set_title(metric_names[metric_idx])
        ax.set_xlabel("Cumulative number of edits")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(f"Accuracy (averaged over each {w}-edit window)")
    fig.suptitle(args.title)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print("Saved plot to", args.output)


if __name__ == "__main__":
    main()