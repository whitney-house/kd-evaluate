"""
Compute a clean quantitative summary from one or more sequential-edit
result files.

Run:
    python scripts/summarize_results.py \
        --run results/sequential_edit_rome_qwen2.5-7b.json "ROME" \
        --run results/sequential_edit_ft_qwen2.5-7b.json "FT-L" \
        --early-late 20 \
        --output results/summary_table.md
"""

import argparse
import json

import numpy as np


def load_metric_series(path):
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "rewrite_acc": [m["post"]["rewrite_acc"][0] for m in metrics],
        "rephrase_acc": [m["post"]["rephrase_acc"][0] for m in metrics],
        "locality_acc": [m["post"]["locality"]["neighborhood_acc"][0] for m in metrics],
    }


def summarize(series, k):
    out = {}
    for name, values in series.items():
        values = np.array(values, dtype=float)
        out[name] = {
            "overall_mean": values.mean(),
            "overall_std": values.std(),
            "first_k_mean": values[:k].mean(),
            "last_k_mean": values[-k:].mean(),
            "drop": values[:k].mean() - values[-k:].mean(),
        }
    return out


def format_markdown_table(all_summaries, k):
    lines = []
    lines.append(f"| Method | Metric | Overall mean (SD) | First {k} mean | Last {k} mean | Drop |")
    lines.append("|---|---|---|---|---|---|")
    for label, summary in all_summaries.items():
        for metric_name, s in summary.items():
            lines.append(
                f"| {label} | {metric_name} | {s['overall_mean']:.2f} ({s['overall_std']:.2f}) "
                f"| {s['first_k_mean']:.2f} | {s['last_k_mean']:.2f} | {s['drop']:+.2f} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("RESULTS_JSON", "LABEL"), required=True)
    parser.add_argument("--early-late", type=int, default=20, help="window size (in edits) for first-K vs last-K comparison")
    parser.add_argument("--output", default="./results/summary_table.md")
    args = parser.parse_args()

    all_summaries = {}
    for path, label in args.run:
        series = load_metric_series(path)
        all_summaries[label] = summarize(series, args.early_late)

    table = format_markdown_table(all_summaries, args.early_late)
    print(table)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nSaved to", args.output)


if __name__ == "__main__":
    main()