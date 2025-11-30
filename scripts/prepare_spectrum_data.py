#!/usr/bin/env python3
"""
Prepare spectrum SFT data:
 - load specified HF dataset(s)
 - map paper domains -> dataset categories
 - robustly extract boxed answers (or fallbacks)
 - optional 10-gram decontamination against provided eval sets
 - output per-domain train & val jsonl files with fields: { "problem", "answer", "source" }
"""
import argparse
import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Set

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------
# Robust boxed-answer extractor
# ---------------------------


def extract_boxed_answer(solution_text: str) -> Optional[str]:
    """
    Conservative extractor for the final boxed answer.
    Returns None if it can't safely find a short answer.
    """
    if not solution_text or not isinstance(solution_text, str):
        return None

    # 1) stack-based parse for last \boxed{...}
    def find_last_boxed(text: str) -> Optional[str]:
        pattern = r"\\boxed\s*\{"
        last_pos = None
        for m in re.finditer(pattern, text):
            last_pos = m.start()
        if last_pos is None:
            return None
        i = last_pos + text[last_pos:].find("{") + 1
        depth = 1
        start = i
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
            i += 1
        return None

    boxed: Optional[str] = find_last_boxed(solution_text)
    if boxed:
        return boxed

    # 2) last inline math ($...$, \(..\), \[..\])
    inline_patterns = [r"\$([^\$]{1,200})\$", r"\\\((.*?)\\\)", r"\\\[(.*?)\\\]"]
    for pat in inline_patterns:
        matches = re.findall(pat, solution_text, flags=re.S)
        if matches:
            candidate: str = matches[-1].strip()
            if 0 < len(candidate) <= 200:
                return candidate

    # 3) last lines with 'Answer', 'ANS', 'Final answer'
    lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip()]
    for ln in reversed(lines[-12:]):
        m = re.search(r"(?i)\b(?:answer|ans|final answer|solution)\b[:\s-]*(.*)", ln)
        if m:
            ans = m.group(1).strip()
            ans = re.sub(r"^\$+|^\(+|^\[+|\\left|\\right", "", ans).strip()
            ans = re.sub(r"\$+$|\)+$|\]+$", "", ans).strip()
            if ans:
                return ans

    # 4) fallback: last numeric-like token / short expression
    tokens = re.findall(
        r"([0-9]+(?:\.[0-9]+)?|-?\d+\/\d+|[A-Za-z0-9\-\+\*/\^\(\)]+)", solution_text
    )
    if tokens:
        cand: str = tokens[-1]
        if len(cand) <= 64:
            return cand

    return None


# ---------------------------
# 10-gram decontamination helpers
# ---------------------------
def build_ngrams(text: str, n: int = 10) -> Set[str]:
    toks = text.split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def compute_eval_ngrams(eval_examples: List[Dict[str, str]], n: int = 10) -> Set[str]:
    s = set()
    for ex in eval_examples:
        prob = ex.get("problem", "") or ""
        s.update(build_ngrams(prob, n=n))
    return s


def filter_by_decontam(
    examples: List[Dict[str, str]], eval_ngrams: Set[str], n: int = 10
) -> List[Dict[str, str]]:
    out = []
    for ex in examples:
        if not (build_ngrams(ex.get("problem", ""), n=n) & eval_ngrams):
            out.append(ex)
    return out


# ---------------------------
# Main processing
# ---------------------------
def prepare(
    hf_id: str,
    output_dir: str,
    domain_map: Optional[Dict[str, List[str]]] = None,
    min_examples: int = 50,
    decontam_eval_ids: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Loading dataset %s", hf_id)
    ds_train = load_dataset(hf_id, split="train")
    ds_test = None
    try:
        ds_test = load_dataset(hf_id, split="test")
    except Exception:
        logging.warning(
            "No test split available for %s; skipping decontamination if requested.",
            hf_id,
        )

    # default domain mapping (paper domains -> dataset categories)
    if domain_map is None:
        domain_map = {
            "algebra": ["Algebra", "Intermediate Algebra", "Algebra"],
            "geometry": ["Geometry"],
            "statistics": ["Counting & Probability", "Probability", "Combinatorics"],
            "calculus": ["Calculus", "Differentiation", "Integration", "Precalculus"],
        }

    # find a field to use for category/type
    candidate_fields = [
        f
        for f in ("type", "category", "problem_type", "domain")
        if f in ds_train.column_names
    ]
    if not candidate_fields:
        raise RuntimeError(
            f"No domain-like field found in {hf_id}. Available: {ds_train.column_names}"
        )
    domain_field = candidate_fields[0]
    logging.info("Using domain field '%s' to filter categories", domain_field)

    # optional evaluation ngram set for decontamination
    eval_ngrams = set()
    if decontam_eval_ids and ds_test is not None:
        eval_examples = []
        for splitid in decontam_eval_ids:
            # Attempt to load splits by name; if not found, use ds_test
            # (The caller can pass paths to other eval datasets if desired)
            # Here we simply use ds_test
            eval_examples.extend(list(ds_test))
        logging.info("Building eval ngrams for decontamination (this may take time)...")
        eval_ngrams = compute_eval_ngrams(eval_examples, n=10)
        logging.info("Eval ngrams set size: %d", len(eval_ngrams))

    summary = {}
    for vt_domain, cats in domain_map.items():
        logging.info("Processing domain '%s' -> categories %s", vt_domain, cats)

        def keep_fn(
            example: Dict[str, str], cats: List[str] = cats, field: str = domain_field
        ) -> bool:
            val = example.get(field, "")
            if isinstance(val, list):
                return any(c in val for c in cats)
            else:
                return any(c in str(val) for c in cats)

        train_filtered = ds_train.filter(keep_fn)
        val_filtered = ds_test.filter(keep_fn) if ds_test is not None else []

        processed_train = []
        for ex in train_filtered:
            sol = ex.get("solution", "") or ex.get("answer", "") or ""
            ans = extract_boxed_answer(sol)
            if not ans:
                # skip examples without safely extracted answer
                continue
            item = {
                "problem": ex.get("problem", "").strip(),
                "answer": ans,
                "source": hf_id,
            }
            processed_train.append(item)

        processed_val = []
        for ex in val_filtered:
            sol = ex.get("solution", "") or ex.get("answer", "") or ""
            ans = extract_boxed_answer(sol)
            if not ans:
                continue
            item = {
                "problem": ex.get("problem", "").strip(),
                "answer": ans,
                "source": hf_id,
            }
            processed_val.append(item)

        # Apply decontamination if requested
        if eval_ngrams:
            before_train = len(processed_train)
            processed_train = filter_by_decontam(processed_train, eval_ngrams)
            logging.info(
                "Decontam: %d -> %d train examples after filtering",
                before_train,
                len(processed_train),
            )

        # fallback warning
        if len(processed_train) < min_examples:
            logging.warning(
                "Only %d train examples for %s (min_examples=%d). Consider adding other sources like NuminaMath.",
                len(processed_train),
                vt_domain,
                min_examples,
            )

        # write jsonl
        out_train = os.path.join(output_dir, f"spectrum_{vt_domain}_train.jsonl")
        out_val = os.path.join(output_dir, f"spectrum_{vt_domain}_val.jsonl")
        with open(out_train, "w", encoding="utf-8") as f:
            for it in processed_train:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        with open(out_val, "w", encoding="utf-8") as f:
            for it in processed_val:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        summary[vt_domain] = {"train": len(processed_train), "val": len(processed_val)}
        logging.info(
            "Saved %s (train=%d val=%d)",
            vt_domain,
            len(processed_train),
            len(processed_val),
        )

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-id", default="hendrycks/competition_math", help="Hugging Face dataset id"
    )
    p.add_argument("--out", default="data", help="output dir")
    p.add_argument("--min-examples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--decontam-eval",
        nargs="*",
        default=None,
        help="(optional) list of eval dataset ids / paths to use for 10-gram decontamination",
    )
    args = p.parse_args()

    summary = prepare(
        hf_id=args.hf_id,
        output_dir=args.out,
        min_examples=args.min_examples,
        decontam_eval_ids=args.decontam_eval,
        seed=args.seed,
    )
    print("Summary:", summary)
