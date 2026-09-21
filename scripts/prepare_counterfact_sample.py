"""
Sample N cases from the official CounterFact dataset (ROME paper format) and
convert them into the flat prompts/subject/target_new/... lists that
editor.edit() expects.

Download the source file first:
    curl -L -o data/counterfact.json https://rome.baulab.info/data/dsets/counterfact.json

Run:
    python prepare_counterfact_sample.py --n 150 --seed 42
"""

import argparse
import json
import random


def convert_case(case):
    rr = case["requested_rewrite"]
    subject = rr["subject"]
    # Original prompts are templates like "The mother tongue of {} is" —
    # fill in the subject to get plain text, matching what editor.edit()
    # (and our hand-written examples) expect.
    prompt = rr["prompt"].format(subject)

    # Use the first paraphrase / first neighborhood prompt — editor.edit()
    # only takes one probe per case (see note in the chat write-up); more
    # cases, not more probes per case, is how we get a stable trend.
    rephrase_prompt = case["paraphrase_prompts"][0] if case["paraphrase_prompts"] else prompt
    neighborhood_prompt = case["neighborhood_prompts"][0] if case["neighborhood_prompts"] else None

    if neighborhood_prompt is None:
        return None  # skip cases with no usable neighborhood probe

    return {
        "case_id": case["case_id"],
        "prompt": prompt,
        "subject": subject,
        "ground_truth": rr["target_true"]["str"],
        "target_new": rr["target_new"]["str"],
        "rephrase_prompt": rephrase_prompt,
        "locality_prompt": neighborhood_prompt,
        # Neighborhood prompts share the *original* fact's target_true —
        # that's the whole point of "neighborhood": these are other
        # subjects that should still map to the same true relation output.
        "locality_ground_truth": rr["target_true"]["str"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/counterfact.json")
    parser.add_argument("--output", default="./data/counterfact_sample.json")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    random.seed(args.seed)
    shuffled = full_dataset[:]
    random.shuffle(shuffled)

    sampled = []
    for case in shuffled:
        converted = convert_case(case)
        if converted is not None:
            sampled.append(converted)
        if len(sampled) >= args.n:
            break

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"Sampled {len(sampled)} cases (seed={args.seed}) -> {args.output}")


if __name__ == "__main__":
    main()