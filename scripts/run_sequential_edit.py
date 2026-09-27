"""
Sequential editing: apply N edits one after another to the SAME model
(not reset between edits), and save the metrics returned at each step.

Run:
    python scripts/run_sequential_edit.py --method ROME --model gpt2-xl
    python scripts/run_sequential_edit.py --method FT --model qwen2.5-7b --examples 5
"""

import argparse
import json
import os

import torch

import easyedit_path  # noqa: F401  (adds EasyEdit/ to sys.path as a side effect)
from easyedit_path import EASYEDIT_DIR
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams, FTHyperParams

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "counterfact_sample.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

HPARAMS_CLASSES = {
    "ROME": ROMEHyperParams,
    "MEMIT": MEMITHyperParams,
    "FT": FTHyperParams,
}


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_text(model, tok, prompt, max_new_tokens=8):
    """Greedy-decode a short continuation for `prompt` on whichever device
    `model` is currently on. Used to show actual model output (not just the
    0/1 accuracy score) for a handful of qualitative examples."""
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def pick_example_indices(n_cases, n_examples):
    """Evenly space example indices across the sequence (first, ~1/4, ~1/2,
    ~3/4, last), so qualitative examples cover early/mid/late edits, not
    just the first few."""
    if n_examples >= n_cases:
        return list(range(n_cases))
    step = (n_cases - 1) / (n_examples - 1) if n_examples > 1 else 0
    return sorted({round(i * step) for i in range(n_examples)})


def collect_generations(model, tok, cases, indices):
    """For each selected case, generate text for its edit prompt, rephrase
    prompt, and locality probe -- this is what actually went into/came out
    of the model, beyond the aggregate accuracy numbers."""
    out = []
    for idx in indices:
        c = cases[idx]
        out.append({
            "case_index": idx,
            "case_id": c["case_id"],
            "prompt": c["prompt"],
            "rephrase_prompt": c["rephrase_prompt"],
            "locality_prompt": c["locality_prompt"],
            "generated_prompt": generate_text(model, tok, c["prompt"]),
            "generated_rephrase": generate_text(model, tok, c["rephrase_prompt"]),
            "generated_locality": generate_text(model, tok, c["locality_prompt"]),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ROME", "MEMIT", "FT"], default="ROME")
    parser.add_argument("--model", default="qwen2.5-7b", help="matches a filename under EasyEdit/hparams/<method>/")
    parser.add_argument("--examples", type=int, default=0,
                         help="if > 0, also save N before/after generation examples spread across the sequence")
    args = parser.parse_args()

    # hparams/data/stats live inside the EasyEdit checkout itself (that's
    # where EasyEdit expects to find and write them); our own sampled
    # edit data and results stay in this repo.
    hparams_path = os.path.join(EASYEDIT_DIR, "hparams", args.method, f"{args.model}.yaml")
    results_path = os.path.join(RESULTS_DIR, f"sequential_edit_{args.method.lower()}_{args.model}.json")
    examples_path = os.path.join(RESULTS_DIR, f"qualitative_examples_{args.method.lower()}_{args.model}.json")

    cases = load_cases(DATA_PATH)

    hparams_cls = HPARAMS_CLASSES[args.method]
    hparams = hparams_cls.from_hparams(hparams_path)
    editor = BaseEditor.from_hparams(hparams)

    example_indices = pick_example_indices(len(cases), args.examples) if args.examples > 0 else []

    # Capture what the UNEDITED model says, before any edits happen.
    before = collect_generations(editor.model, editor.tok, cases, example_indices) if example_indices else []

    prompts = [c["prompt"] for c in cases]
    subject = [c["subject"] for c in cases]
    ground_truth = [c["ground_truth"] for c in cases]
    target_new = [c["target_new"] for c in cases]
    rephrase_prompts = [c["rephrase_prompt"] for c in cases]
    locality_inputs = {
        "neighborhood": {
            "prompt": [c["locality_prompt"] for c in cases],
            "ground_truth": [c["locality_ground_truth"] for c in cases],
        }
    }

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subject,
        ground_truth=ground_truth,
        target_new=target_new,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        sequential_edit=True,
        keep_original_weight=False,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[{args.method} / {args.model}] Sequential edit complete ({len(cases)} edits). Saved to:", results_path)

    print("\nstep  rewrite_acc  rephrase_acc  locality_acc")
    for i, m in enumerate(metrics):
        post = m["post"]
        rewrite = post["rewrite_acc"][0]
        rephrase = post["rephrase_acc"][0]
        locality = post["locality"]["neighborhood_acc"][0]
        print(f"{i:>4}  {rewrite:>11.2f}  {rephrase:>12.2f}  {locality:>13.2f}")

    if example_indices:
        # Capture what the model says AFTER all sequential edits, for the
        # same prompts -- editor.model holds the fully-edited weights.
        after = collect_generations(editor.model, editor.tok, cases, example_indices)

        combined = []
        for b, a, idx in zip(before, after, example_indices):
            c = cases[idx]
            combined.append({
                "case_index": idx,
                "case_id": c["case_id"],
                "subject": c["subject"],
                "target_new": c["target_new"],
                "prompt": c["prompt"],
                "before_edit": b["generated_prompt"],
                "after_edit": a["generated_prompt"],
                "rephrase_prompt": c["rephrase_prompt"],
                "rephrase_before": b["generated_rephrase"],
                "rephrase_after": a["generated_rephrase"],
                "locality_prompt": c["locality_prompt"],
                "locality_ground_truth": c["locality_ground_truth"],
                "locality_before": b["generated_locality"],
                "locality_after": a["generated_locality"],
            })

        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(combined)} qualitative examples to:", examples_path)
        print("\n--- Qualitative examples ---")
        for ex in combined:
            print(f"\n[case {ex['case_index']}] {ex['prompt']}")
            print(f"  target_new     : {ex['target_new']}")
            print(f"  before edit    : {ex['before_edit']}")
            print(f"  after edit     : {ex['after_edit']}")
            print(f"  rephrase       : {ex['rephrase_prompt']}")
            print(f"    before -> after : {ex['rephrase_before']!r} -> {ex['rephrase_after']!r}")
            print(f"  locality probe : {ex['locality_prompt']} (expected: {ex['locality_ground_truth']})")
            print(f"    before -> after : {ex['locality_before']!r} -> {ex['locality_after']!r}")


if __name__ == "__main__":
    main()