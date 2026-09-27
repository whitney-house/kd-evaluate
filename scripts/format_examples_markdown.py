"""
Turn a qualitative_examples_*.json file (produced by run_sequential_edit.py
with --examples N) into a readable Markdown table for the report.

Run:
    python scripts/format_examples_markdown.py \
        --input results/qualitative_examples_rome_qwen2.5-7b.json \
        --output results/qualitative_examples_rome_qwen2.5-7b.md
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        examples = json.load(f)

    lines = ["| Case | Prompt | Target | Before edit | After edit | Locality probe | Locality before -> after |",
             "|---|---|---|---|---|---|---|"]
    for ex in examples:
        lines.append(
            f"| {ex['case_index']} | {ex['prompt']} | {ex['target_new']} "
            f"| {ex['before_edit']!r} | {ex['after_edit']!r} "
            f"| {ex['locality_prompt']} | {ex['locality_before']!r} -> {ex['locality_after']!r} |"
        )

    table = "\n".join(lines)
    print(table)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nSaved to", args.output)


if __name__ == "__main__":
    main()