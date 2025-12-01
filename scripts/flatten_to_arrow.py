#!/usr/bin/env python3
"""
scripts/flatten_to_arrow.py

- Reads flat/nested JSONL produced by prepare_spectrum_data.py
  (expects fields like: problem, solution, answer, domain, teacher_meta, verified)
- Produces a Hugging Face Dataset saved with `save_to_disk()` (Arrow)
- Default tokenization max_length=4096
- Optionally mask prompt tokens in labels (labels=-100)
- Optionally filter to verified-only examples
- Batched tokenization (num_proc / batch_size supported)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# optional imports at runtime
try:
    from datasets import Dataset
except Exception:
    Dataset = None  # checked later

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore


# -------------------------
# Utilities (same logic as prepare)
# -------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def fingerprint(problem: str, solution: str) -> str:
    key = (normalize_text(problem) + "||" + normalize_text(solution)).encode("utf-8")
    import hashlib

    return hashlib.sha1(key).hexdigest()


def iter_input_records(paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    logging.warning("Skipping invalid JSON line in %s: %s", p, e)
                    continue


def extract_entries_from_record(
    rec: Dict[str, Any],
) -> Iterable[tuple[str, str, Optional[str], bool, Dict[str, Any]]]:
    """
    Normalize both flat and nested records into a list of tuples:
      (problem, solution, answer, verified, teacher_meta)
    """
    problem = rec.get("problem") or rec.get("question") or ""
    # If nested teacher outputs:
    if "teacher_outputs" in rec and isinstance(rec["teacher_outputs"], list):
        for out in rec["teacher_outputs"]:
            sol = out.get("solution") or out.get("answer") or ""
            ans = out.get("answer") or None
            verified = bool(out.get("verified", False)) or bool(out.get("verify_info"))
            meta = out.get("meta") or {}
            yield (problem, sol, ans, verified, meta)
    # If flat record containing solution and possibly teacher_meta:
    elif "solution" in rec or "answer" in rec:
        sol = rec.get("solution") or rec.get("answer") or ""
        ans = rec.get("answer") or None
        verified = bool(rec.get("verified", False))
        meta = rec.get("teacher_meta") or {}
        yield (problem, sol, ans, verified, meta)
    else:
        # Fallback: keys look different, try any plausible fields
        sol = rec.get("text") or rec.get("completion") or ""
        ans = rec.get("answer") or None
        verified = bool(rec.get("verified", False))
        yield (problem, sol, ans, verified, {})


# -------------------------
# Build prompt/completion
# -------------------------
def build_prompt(problem: str, prompt_template: str) -> str:
    return prompt_template.replace("{problem}", problem)


def build_completion(
    solution: str, answer: Optional[str], keep_cot: bool, answer_prefix: str
) -> str:
    if keep_cot and solution:
        return solution.strip()
    if answer:
        return f"{answer_prefix}{answer}".strip()
    return solution.strip() if solution else ""


# -------------------------
# Collector (dedupe, filter, shuffle)
# -------------------------
def collect_examples(
    input_paths: List[str],
    prompt_template: str,
    keep_cot: bool,
    answer_prefix: str,
    dedupe: bool,
    shuffle: bool,
    seed: int,
    max_examples: Optional[int],
    verified_only: bool,
) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    total_in = 0
    for rec in iter_input_records(input_paths):
        for problem, solution, answer, verified, meta in extract_entries_from_record(
            rec
        ):
            total_in += 1
            problem_norm = normalize_text(problem)
            solution_norm = normalize_text(solution)
            if not problem_norm:
                continue
            # If verified_only is True, skip unverified
            if verified_only and not verified:
                continue
            # If both solution and answer missing, skip
            if (not solution_norm) and (not answer):
                continue
            completion = build_completion(
                solution_norm, answer, keep_cot, answer_prefix
            )
            if not completion:
                continue
            fp = fingerprint(problem_norm, completion)
            if dedupe and fp in seen:
                continue
            seen.add(fp)
            prompt = build_prompt(problem_norm, prompt_template)
            out.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "problem": problem_norm,
                    "solution": solution_norm,
                    "answer": answer,
                    "verified": bool(verified),
                    "teacher_meta": meta,
                }
            )
            if max_examples and len(out) >= max_examples:
                break
        if max_examples and len(out) >= max_examples:
            break
    logging.info("Collected %d examples (from %d input records)", len(out), total_in)
    if shuffle:
        random.seed(seed)
        random.shuffle(out)
    return out


# -------------------------
# Tokenization & labels prepare (batched)
# -------------------------
def tokenize_and_prepare_labels_hf(
    examples: Dict[str, List[Any]],
    tokenizer: Any,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    text_field: str = "text",
    delim: str = "\n\n### Response:\n\n",
    max_length: int = 4096,
    padding: str = "max_length",
    mask_prompt: bool = True,
) -> Dict[str, Any]:
    """
    examples: a batch dict with lists for 'prompt' and 'completion' (and possibly 'problem', metadata).
    Returns dict of tokenized fields with 'input_ids', 'attention_mask', 'labels'.
    """
    prompts = examples[prompt_field]
    completions = examples[completion_field]
    # build the text: prompt + delim + completion
    texts = [p + delim + c for p, c in zip(prompts, completions)]
    # tokenize full text
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=padding,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", [[1] * len(ids) for ids in input_ids])

    # compute prompt token lengths per example (without padding)
    prompt_enc = tokenizer(
        prompts, truncation=True, max_length=max_length, padding=False
    )
    prompt_lens = [len(x) for x in prompt_enc["input_ids"]]

    labels_batch = []
    for ids, plen in zip(input_ids, prompt_lens):
        labels = ids.copy()
        if mask_prompt:
            # set prompt positions to -100 up to plen or len(labels)
            for i in range(min(plen, len(labels))):
                labels[i] = -100
        labels_batch.append(labels)

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_batch,
    }
    return out


# -------------------------
# Main: build HF Dataset and optionally tokenize
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Flatten JSONL -> HF Dataset (Arrow). Patched defaults for VibeThinker."
    )
    p.add_argument(
        "--inputs", nargs="+", required=True, help="Input JSONL files (flat or nested)"
    )
    p.add_argument(
        "--out-dir", required=True, help="Output dir to save HF dataset (save_to_disk)"
    )
    p.add_argument(
        "--format",
        choices=["trl", "plain"],
        default="trl",
        help="TRL formatting or plain prompt/completion",
    )
    p.add_argument(
        "--prompt-template",
        default="Problem: {problem}\nSolution:",
        help="Prompt template; include {problem}",
    )
    p.add_argument(
        "--keep-cot",
        action="store_true",
        help="Include full CoT solution in completion",
    )
    p.add_argument(
        "--answer-prefix",
        default="Answer: ",
        help="If not keep_cot, prefix used when using final answer",
    )
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate by problem+completion fingerprint",
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle examples before writing"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit total examples after dedupe/shuffle",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer model id (tokenize and save input_ids/labels)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Tokenization max length (default 4096 for long CoT)",
    )
    p.add_argument("--padding", choices=["max_length", "longest"], default="max_length")
    p.add_argument(
        "--mask-prompt",
        action="store_true",
        help="Mask prompt tokens in labels (set to -100). Recommended for SFT.",
    )
    p.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="num_proc for dataset.map tokenization",
    )
    p.add_argument(
        "--batch-size", type=int, default=128, help="Batched tokenization batch size"
    )
    p.add_argument(
        "--verified-only",
        action="store_true",
        help="Include only examples marked verified",
    )
    args = p.parse_args()

    if Dataset is None:
        logging.error("datasets package is required: pip install datasets")
        return
    if args.tokenizer and AutoTokenizer is None:
        logging.error(
            "transformers package is required for tokenization: pip install transformers"
        )
        return

    examples = collect_examples(
        input_paths=args.inputs,
        prompt_template=args.prompt_template,
        keep_cot=args.keep_cot,
        answer_prefix=args.answer_prefix,
        dedupe=args.dedupe,
        shuffle=args.shuffle,
        seed=args.seed,
        max_examples=args.max_examples,
        verified_only=args.verified_only,
    )

    if len(examples) == 0:
        logging.error("No examples extracted. Exiting.")
        return

    # Convert to Dataset
    # We'll keep columns: prompt, completion, problem, solution, answer, verified, teacher_meta
    ds_dict = {
        "prompt": [e["prompt"] for e in examples],
        "completion": [e["completion"] for e in examples],
        "problem": [e["problem"] for e in examples],
        "solution": [e["solution"] for e in examples],
        "answer": [e["answer"] for e in examples],
        "verified": [e["verified"] for e in examples],
        "teacher_meta": [e.get("teacher_meta", {}) for e in examples],
    }

    ds = Dataset.from_dict(ds_dict)
    logging.info(
        "Built raw Dataset with %d examples. Columns: %s", len(ds), ds.column_names
    )

    # If tokenizer provided -> tokenize and create labels
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)  # type: ignore
        # Ensure pad token exists
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        # Prepare text column used for tokenization (prompt + delim + completion)
        delim = "\n\n### Response:\n\n"
        ds = ds.map(
            lambda batch: {
                "text": [
                    p + delim + c for p, c in zip(batch["prompt"], batch["completion"])
                ]
            },
            batched=True,
            batch_size=args.batch_size,
            remove_columns=[],
        )

        # Tokenize in batches; use map with batched and num_proc if provided
        def hf_map_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            # batch is a dict of lists with key "prompt","completion","text"
            tokenized = tokenize_and_prepare_labels_hf(
                examples=batch,
                tokenizer=tokenizer,
                prompt_field="prompt",
                completion_field="completion",
                text_field="text",
                delim=delim,
                max_length=args.max_length,
                padding=args.padding,
                mask_prompt=args.mask_prompt,
            )
            # Keep metadata columns in output mapping (they are present in batch)
            out = {}
            out.update(tokenized)
            # do not remove other metadata columns in this mapping stage; datasets.map will merge them
            return out

        # apply tokenization mapping (we keep metadata columns)
        ds_tokenized = ds.map(
            hf_map_batch,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=ds.column_names,  # remove old text/prompt/completion columns (labels/input_ids will replace)
        )

        # Save tokenized dataset
        os.makedirs(args.out_dir, exist_ok=True)
        logging.info("Saving tokenized dataset to %s", args.out_dir)
        ds_tokenized.save_to_disk(args.out_dir)
        logging.info("Saved tokenized dataset. Features: %s", ds_tokenized.features)
    else:
        # Save raw text dataset
        os.makedirs(args.out_dir, exist_ok=True)
        logging.info("Saving raw text dataset to %s", args.out_dir)
        ds.save_to_disk(args.out_dir)
        logging.info("Saved. Columns: %s", ds.column_names)

    logging.info("Done. Wrote %d examples to %s", len(examples), args.out_dir)


if __name__ == "__main__":
    main()
