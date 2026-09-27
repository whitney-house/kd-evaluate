"""
Plot the full 2x2 design (2 methods x 2 models) as a grid: rows = model,
columns = metric, with ROME and FT-L overlaid as two lines in each subplot.

Run:
    python scripts/plot_2x2_grid.py --window 10
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "./results"
MODELS = ["qwen2.5-7b", "gpt2-xl"]
MODEL_LABELS = {"qwen2.5-7b": "Qwen2.5-7B", "gpt2-xl": "GPT2-XL"}
METHODS = ["rome", "ft"]
METHOD_LABELS = {"rome": "ROME", "ft": "FT-L"}
METHOD_COLORS = {"rome": "tab:blue", "ft": "tab:red"}
METRIC_KEYS = ["rewrite_acc", "rephrase_acc", "locality_acc"]
METRIC_TITLES = ["Reliability (rewrite_acc)", "Generalization (rephrase_acc)", "Locality (neighborhood_acc)"]


def windowed_mean(values, window):
    values = np.array(values, dtype=float)
    n_windows = len(values) // window
    trimmed = values[: n_windows * window]
    return trimmed.reshape(n_windows, window).mean(axis=1)


def load_metric_series(path):
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "rewrite_acc": [m["post"]["rewrite_acc"][0] for m in metrics],
        "rephrase_acc": [m["post"]["rephrase_acc"][0] for m in metrics],
        "locality_acc": [m["post"]["locality"]["neighborhood_acc"][0] for m in metrics],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output", default="./results/grid_2x2_comparison.png")
    args = parser.parse_args()
    w = args.window

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)

    for row, model in enumerate(MODELS):
        for col, metric_key in enumerate(METRIC_KEYS):
            ax = axes[row][col]
            for method in METHODS:
                path = f"{RESULTS_DIR}/sequential_edit_{method}_{model}.json"
                series = load_metric_series(path)[metric_key]
                series_w = windowed_mean(series, w)
                x = [(i + 1) * w for i in range(len(series_w))]
                ax.plot(x, series_w, marker="o", label=METHOD_LABELS[method], color=METHOD_COLORS[method])
            if row == 0:
                ax.set_title(METRIC_TITLES[col])
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nAccuracy (window={w})")
            if row == 1:
                ax.set_xlabel("Cumulative number of edits")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle("ROME vs FT-L x Qwen2.5-7B vs GPT2-XL (sequential editing, 150 CounterFact cases)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print("Saved plot to", args.output)


if __name__ == "__main__":
    main()