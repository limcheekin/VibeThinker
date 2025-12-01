#!/usr/bin/env python3
"""
Patched prepare_spectrum_data.py

- Adds semantic deduplication using sentence-transformers (all-MiniLM-L6-v2).
- Verification priority: gold -> sympy -> consensus (two-teacher calls).
- Keeps resume/checkpoint behavior and teacher_records.jsonl for resumability.
- Usage notes at bottom.

Install:
  pip install sentence-transformers
  pip install sympy            # optional but recommended
  pip install openai           # if using openai backend
  pip install transformers accelerate  # if using hf backend
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from datasets import load_dataset
from tqdm import tqdm

if TYPE_CHECKING:
    import openai as openai_module
    import sympy
    import torch as torch_module
    from sentence_transformers import SentenceTransformer as SentenceTransformerType
    from sentence_transformers import util as sbert_util_module
    from transformers import Pipeline
else:
    sympy = None
    openai_module = None
    torch_module = None
    Pipeline = None
    SentenceTransformerType = None
    sbert_util_module = None

# Optional libs
try:
    import sympy as sp
except Exception:
    sp = None

try:
    import openai
except Exception:
    openai = None  # type: ignore[assignment]

try:
    import torch
    from transformers import pipeline as hf_pipeline_constructor
except Exception:
    torch = None  # type: ignore[assignment]
    hf_pipeline_constructor = None  # type: ignore[assignment]

# Semantic dedupe libs
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as sbert_util
except Exception:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    sbert_util = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------
# Text normalization / small helpers
# ---------------------------
def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = s.replace("\r\n", "\n")
    s = html.unescape(s)
    s = " ".join(s.split())
    return s


def text_fingerprint(s: str) -> str:
    return hashlib.sha1(normalize_text(s).encode("utf-8")).hexdigest()


# ---------------------------
# Boxed answer extraction (robust)
# ---------------------------
def extract_boxed_answer(solution_text: str) -> Optional[str]:
    if not solution_text or not isinstance(solution_text, str):
        return None

    # Stack-aware \boxed{...}
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

    boxed = find_last_boxed(solution_text)
    if boxed:
        return boxed

    inline_patterns = [r"\$([^\$]{1,200})\$", r"\\\((.*?)\\\)", r"\\\[(.*?)\\\]"]
    for pat in inline_patterns:
        matches = re.findall(pat, solution_text, flags=re.S)
        if matches:
            candidate = str(matches[-1]).strip()
            if 0 < len(candidate) <= 200:
                return candidate

    lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip()]
    for ln in reversed(lines[-12:]):
        m = re.search(r"(?i)\b(?:answer|ans|final answer|solution)\b[:\s-]*(.*)", ln)
        if m:
            ans = m.group(1).strip()
            ans = re.sub(r"^\$+|^\(+|^\[+|\\left|\\right", "", ans).strip()
            ans = re.sub(r"\$+$|\)+$|\]+$", "", ans).strip()
            if ans:
                return ans

    tokens = re.findall(
        r"([0-9]+(?:\.[0-9]+)?|-?\d+\/\d+|[A-Za-z0-9\-\+\*/\^\(\)]+)", solution_text
    )
    if tokens:
        cand = str(tokens[-1])
        if len(cand) <= 64:
            return cand

    return None


# ---------------------------
# Verification helpers
# ---------------------------

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:
    sp = None
    parse_expr = None
    standard_transformations = None
    implicit_multiplication_application = None

_NUMERIC_TOL = 1e-9


def _latex_to_ascii(s: str) -> str:
    r"""
    Best-effort convert a few LaTeX constructs to ascii math SymPy can parse.
    - \frac{a}{b} -> (a)/(b)
    - remove $ ... $ and \( \) \[ \] wrappers
    - convert common \cdot, \times to *
    - naive handling of superscript a^{b} -> a**(b)
    """
    if not s:
        return s
    t = s
    # remove wrappers
    t = re.sub(r"\$\s*", "", t)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", t)

    # \frac{num}{den} -> (num)/(den) (handles nested braces greedily but not arbitrarily complex TeX)
    def frac_repl(m: re.Match[str]) -> str:
        return f"({m.group(1)})/({m.group(2)})"

    t = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", frac_repl, t)

    # simple superscripts a^{b} -> a**(b)
    t = re.sub(r"([A-Za-z0-9\)\]])\s*\^\s*\{\s*([^{}]+)\s*\}", r"\1**(\2)", t)
    t = re.sub(r"([A-Za-z0-9\)\]])\s*\^\s*([0-9]+)", r"\1**(\2)", t)

    # Replace \times, \cdot with *
    t = re.sub(r"\\times|\\cdot", "*", t)

    # remove leftover TeX commands like \left \right
    t = re.sub(r"\\[a-zA-Z]+", "", t)

    # collapse whitespace
    t = " ".join(t.split())
    return t


def _try_sympy_equal(a: str, b: str) -> Optional[bool]:
    """
    Try symbolic equality using sympy. Returns True/False on success, or None if SymPy can't decide/parse.
    """
    if sp is None or parse_expr is None:
        return None
    try:
        # convert LaTeX -> ascii then parse
        a_conv = _latex_to_ascii(a)
        b_conv = _latex_to_ascii(b)

        # use safe transformations
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        a_expr = parse_expr(a_conv, transformations=transformations)
        b_expr = parse_expr(b_conv, transformations=transformations)

        diff = sp.simplify(a_expr - b_expr)
        # If sympy returns zero symbolic, they're equal
        if diff == 0:
            return True
        # else if numeric evaluate small
        try:
            diff_num = float(sp.N(diff))
            return abs(diff_num) <= _NUMERIC_TOL
        except Exception:
            return False
    except Exception:
        return None


def _try_numeric_equal(a: str, b: str) -> Optional[bool]:
    """
    Numeric (and complex) compare:
    - handle fractions "1/2" -> 0.5
    - handle decimals and integers
    - handle complex forms '3+4i' or '3+4j'
    Returns True/False, or None if we can't extract numbers.
    """
    if not a or not b:
        return None

    def extract_number(text: str) -> Optional[complex]:
        t = _latex_to_ascii(text)  # reuse latex -> ascii helper
        t = t.strip()
        # find first fraction token like -?num/den
        frac_m = re.search(r"(-?\d+\s*/\s*-?\d+)", t)
        if frac_m:
            tok = frac_m.group(1).replace(" ", "")
            try:
                num, den = tok.split("/")
                return complex(float(num)) / complex(float(den))
            except Exception:
                return None
        # try to find explicit complex like a+bi or a+bj
        # normalize 'i' -> 'j' for python complex parsing but only when it looks numeric
        # remove spaces around +/-
        t_norm = re.sub(r"\s+", "", t)
        # convert trailing 'i' tokens to 'j' (careful: this is for numeric patterns)
        t_norm = re.sub(r"([0-9\)\]])i\b", r"\1j", t_norm)
        t_norm = t_norm.replace("i+", "j+").replace("i-", "j-")
        # try complex() parse
        try:
            return complex(t_norm)
        except Exception:
            # fallback: extract first decimal/integer token
            m = re.search(r"(-?\d+\.\d+|-?\d+)", t)
            if not m:
                return None
            try:
                return complex(float(m.group(1)))
            except Exception:
                return None

    na = extract_number(a)
    nb = extract_number(b)
    if na is None or nb is None:
        return None

    # compare by absolute difference for complex numbers
    diff = abs(na - nb)
    tol = max(_NUMERIC_TOL, 1e-6 * max(abs(na), abs(nb), 1.0))
    return bool(diff <= tol)


def verify_against_gold(
    extracted_answer: Optional[str], gold_answer: Optional[str]
) -> bool:
    """
    Robust verification:
    1) fast normalized string match
    2) sympy-based equivalence (after LaTeX -> ascii normalization)
    3) numeric fallback (fractions/decimals) with tolerance
    """
    if not extracted_answer or not gold_answer:
        return False

    def norm_str(x: str) -> str:
        return re.sub(r"\\boxed\{|\}|\s+|,", "", str(x)).strip().lower()

    if norm_str(extracted_answer) == norm_str(gold_answer):
        return True

    # sympy attempt
    sym_res = _try_sympy_equal(extracted_answer, gold_answer)
    if sym_res is True:
        return True
    if sym_res is False:
        return False

    # numeric fallback
    num_res = _try_numeric_equal(extracted_answer, gold_answer)
    if num_res is True:
        return True
    if num_res is False:
        return False

    # last resort: try normalizing LaTeX to ascii and compare
    if (
        _latex_to_ascii(extracted_answer).strip().lower()
        == _latex_to_ascii(gold_answer).strip().lower()
    ):
        return True

    return False


def verify_answer_via_sympy(
    problem: str, solution_text: str, claimed_answer: Optional[str]
) -> Dict[str, Any]:
    """Best-effort SymPy-based verification; returns dict with 'verified' bool and 'reason'."""
    if sp is None:
        return {"verified": False, "reason": "sympy_not_installed"}
    if not claimed_answer:
        return {"verified": False, "reason": "no_claimed_answer"}
    try:
        nums = re.findall(r"-?\d+(?:\.\d+)?", claimed_answer)
        if not nums:
            try:
                _ = sp.N(sp.sympify(claimed_answer))
                return {"verified": True, "reason": "evaluated_claimed_answer"}
            except Exception:
                return {"verified": False, "reason": "no_numeric_in_claimed_answer"}
        val = float(nums[-1])
        eqs = re.findall(
            r"([A-Za-z0-9\+\-\*/\^\(\)\s]+=[A-Za-z0-9\+\-\*/\^\(\)\s]+)", problem
        )
        for eqtext in eqs[:2]:
            try:
                left, right = eqtext.split("=")
                x = sp.symbols("x")
                lv = sp.sympify(left).subs(x, val)
                rv = sp.sympify(right).subs(x, val)
                if sp.simplify(lv - rv) == 0:
                    return {"verified": True, "reason": "equation_holds_substitution"}
            except Exception:
                continue
        try:
            _ = sp.N(sp.sympify(claimed_answer))
            return {"verified": True, "reason": "evaluated_claimed_answer"}
        except Exception:
            return {"verified": False, "reason": "no_match_found"}
    except Exception as e:
        return {"verified": False, "reason": f"exception:{e}"}


# ---------------------------
# Teacher backends (OpenAI / HF)
# ---------------------------
def call_teacher_openai(
    prompt: str,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    n: int = 1,
) -> List[str]:
    if openai is None:
        raise RuntimeError("openai package not installed")
    for attempt in range(5):
        try:
            resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            outs = []
            for ch in resp.get("choices", []):
                text = (
                    ch["message"]["content"] if "message" in ch else ch.get("text", "")
                )
                outs.append(text)
            return outs
        except Exception as e:
            logging.warning("OpenAI attempt %d failed: %s", attempt + 1, e)
            time.sleep(min(10, 2**attempt))
    raise RuntimeError("OpenAI teacher failed after retries")


def init_hf_pipeline(model_id: str) -> Any:
    if hf_pipeline_constructor is None:
        raise RuntimeError("transformers pipeline not available")
    device = 0 if torch and torch.cuda.is_available() else -1
    gen = hf_pipeline_constructor("text-generation", model=model_id, device=device)
    return gen


def call_teacher_hf(
    gen_pipeline_obj: Any,
    prompt: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
) -> List[str]:
    for attempt in range(3):
        try:
            out = gen_pipeline_obj(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1,
            )
            if isinstance(out, list):
                return [o.get("generated_text", o.get("text", "")) for o in out]
            return [str(out)]
        except Exception as e:
            logging.warning("HF pipeline attempt %d failed: %s", attempt + 1, e)
            time.sleep(1.5**attempt)
    return []


# ---------------------------
# Semantic deduplication (SBERT)
# ---------------------------
def init_sbert(model_name: str = "all-MiniLM-L6-v2") -> Any:
    if SentenceTransformer is None or sbert_util is None:
        raise RuntimeError(
            "sentence-transformers not installed. pip install sentence-transformers"
        )
    return SentenceTransformer(model_name)


def is_semantically_distinct(
    candidate: str, accepted_texts: List[str], model: Any, threshold: float = 0.90
) -> bool:
    """
    Return True if candidate is semantically distinct from all accepted_texts.
    Uses cosine similarity between SBERT embeddings.
    """
    if not accepted_texts:
        return True
    emb_c = model.encode(candidate, convert_to_tensor=True)
    embs = model.encode(accepted_texts, convert_to_tensor=True)
    sims = sbert_util.cos_sim(emb_c, embs).cpu().numpy().flatten()
    max_sim = float(sims.max()) if len(sims) > 0 else 0.0
    return max_sim < threshold


# ---------------------------
# Distill multiple solutions with verification and semantic dedupe
# ---------------------------
def distill_multiple_solutions(
    problem: str,
    gold_solution: Optional[str],
    teacher_backend: str,
    teacher_model: str,
    n_solutions_target: int = 3,
    sampling_configs: Optional[List[Dict[str, Any]]] = None,
    verify: bool = True,
    max_attempts: int = 12,
    hf_pipeline_obj: Any = None,
    sbert_model: Any = None,
    sim_threshold: float = 0.90,
    secondary_teacher_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate up to n_solutions_target diverse solutions for `problem`.
    Verification priority:
      1) verify_against_gold (if gold_solution provided)
      2) SymPy verification (if available)
      3) Two-teacher consensus: generate additional outputs from a secondary teacher or second call
    Semantic deduplication via sbert_model when provided.
    """
    if sampling_configs is None:
        sampling_configs = [
            {"temperature": 1.0, "top_p": 0.95},
            {"temperature": 0.9, "top_p": 0.95},
            {"temperature": 1.1, "top_p": 0.98},
            {"temperature": 0.7, "top_p": 0.9},
            {"temperature": 1.3, "top_p": 0.98},
        ]

    results: List[Dict[str, Any]] = []
    seen_fps: Set[str] = set()
    attempts = 0
    cfg_idx = 0

    method_guidance = [
        "Provide a full chain-of-thought solution and end with \\boxed{...} for the final numeric answer.",
        "Provide a concise solution focusing on algebraic manipulation; box the final answer.",
        "Solve by substitution or elimination; show steps and box the final answer.",
        "Give a high-level strategy then the detailed step-by-step solution; box the final answer.",
        "Try a factoring-based approach if applicable; show steps and box the final answer.",
    ]

    # helper to call primary teacher with a sampling cfg
    def call_primary(prompt: str, cfg: Dict[str, Any]) -> List[str]:
        if teacher_backend == "openai":
            return call_teacher_openai(
                prompt,
                model=teacher_model,
                temperature=cfg["temperature"],
                max_tokens=1024,
                n=1,
            )
        else:
            return call_teacher_hf(
                hf_pipeline_obj,
                prompt,
                temperature=cfg["temperature"],
                top_p=cfg.get("top_p", 0.95),
            )

    # helper to call secondary teacher or re-sample
    def call_secondary(prompt: str) -> List[str]:
        if secondary_teacher_model:
            if teacher_backend == "openai":
                return call_teacher_openai(
                    prompt,
                    model=secondary_teacher_model,
                    temperature=1.0,
                    max_tokens=1024,
                    n=1,
                )
            else:
                # use another HF pipeline if model differs
                try:
                    sec_pipe = init_hf_pipeline(secondary_teacher_model)
                    return call_teacher_hf(sec_pipe, prompt, temperature=1.0)
                except Exception:
                    # fallback to same pipeline resampling
                    return call_teacher_hf(hf_pipeline_obj, prompt, temperature=1.1)
        else:
            # re-sample with slightly different cfg
            return call_primary(prompt, {"temperature": 1.1, "top_p": 0.98})

    while len(results) < n_solutions_target and attempts < max_attempts:
        cfg = sampling_configs[cfg_idx % len(sampling_configs)]
        cfg_idx += 1
        attempts += 1
        prompt = f"{method_guidance[(attempts - 1) % len(method_guidance)]}\nProblem: {problem}\nSolution:"

        try:
            out_texts = call_primary(prompt, cfg)
        except Exception as e:
            logging.warning("Primary teacher call failed: %s", e)
            time.sleep(0.5)
            continue

        for txt in out_texts:
            txt_norm = normalize_text(txt)
            fp = text_fingerprint(txt_norm)
            if fp in seen_fps:
                continue
            seen_fps.add(fp)

            boxed = extract_boxed_answer(txt_norm)
            verified = False
            verify_info = {"method": None, "details": None}

            # 1) verify against gold if available
            if gold_solution and boxed:
                if verify_against_gold(boxed, gold_solution):
                    verified = True
                    verify_info = {
                        "method": "gold",  # type: ignore[dict-item]
                        "details": "exact_or_normalized_match",  # type: ignore[dict-item]
                    }
            # 2) SymPy verification
            if not verified and verify and boxed:
                sym = verify_answer_via_sympy(problem, txt_norm, boxed)
                if sym.get("verified", False):
                    verified = True
                    verify_info = {"method": "sympy", "details": sym.get("reason")}  # type: ignore[dict-item]

            # 3) consensus fallback: require agreement between two teacher draws (or secondary teacher)
            if not verified and verify:
                # get secondary output
                try:
                    sec_outs = call_secondary(prompt)
                except Exception:
                    sec_outs = []
                # extract boxed answers from secondary outs
                sec_answers = []
                for so in sec_outs:
                    so_norm = normalize_text(so)
                    sa = extract_boxed_answer(so_norm)
                    if not sa:
                        # fallback numeric token
                        m = re.findall(r"(-?\d+(?:\.\d+)?|-?\d+\/\d+)", so_norm)
                        sa = m[-1] if m else None
                    if sa:
                        sec_answers.append(sa.strip())
                # consensus if boxed equals any of sec_answers
                if boxed and boxed.strip() in sec_answers and boxed.strip() != "":
                    verified = True
                    verify_info = {
                        "method": "consensus_secondary",  # type: ignore[dict-item]
                        "details": f"matched_secondary:{sec_answers}",  # type: ignore[dict-item]
                    }

            # Semantic dedupe: ensure candidate is semantically distinct relative to already accepted solutions
            accepted_solutions = [r["solution"] for r in results]
            semantically_distinct = True
            if sbert_model is not None and boxed:
                try:
                    semantically_distinct = is_semantically_distinct(
                        txt_norm,
                        accepted_solutions,
                        sbert_model,
                        threshold=sim_threshold,
                    )
                except Exception as e:
                    logging.warning("Semantic dedupe check failed: %s", e)
                    semantically_distinct = True  # fail open for safety

            # append candidate with metadata; trust verified flag but still store unverified ones (marked)
            results.append(
                {
                    "solution": txt_norm,
                    "answer": boxed,
                    "fingerprint": fp,
                    "verified": verified,
                    "verify_info": verify_info,
                    "semantic_distinct": semantically_distinct,
                    "meta": {
                        "teacher": teacher_model,
                        "sampling_cfg": cfg,
                        "attempt": attempts,
                    },
                }
            )

            # If semantic dedupe is False, mark it but do not count it toward target
            if not semantically_distinct:
                # don't count this as a distinct solution
                continue

            # count only truly distinct accepted (even if unverified, they may be post-processed or reviewed)
            distinct_accepted = [r for r in results if r["semantic_distinct"]]
            if len(distinct_accepted) >= n_solutions_target:
                break

        time.sleep(0.15)

    # final filter: keep only semantically distinct entries, and mark verification status
    final = [r for r in results if r["semantic_distinct"]]
    # ensure deterministic ordering
    return final[:n_solutions_target]


# ---------------------------
# 10-gram decontamination helpers (same concept as before)
# ---------------------------
def build_ngrams(text: str, n: int = 10) -> Set[str]:
    toks = (text or "").split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def compute_eval_ngrams_from_datasets(eval_ids: List[str]) -> Set[str]:
    s = set()
    for dsid in eval_ids:
        # try HF dataset first
        try:
            ds = load_dataset(dsid, split="train")
            for ex in ds:
                prob = ex.get("problem", "") or ex.get("question", "")
                s.update(build_ngrams(prob, 10))
            continue
        except Exception:
            pass
        # fallback: treat as path to jsonl file
        try:
            with open(dsid, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    prob = obj.get("problem") or obj.get("question") or ""
                    s.update(build_ngrams(prob, 10))
        except Exception as e:
            logging.warning("Could not load eval id/path %s: %s", dsid, e)
            continue
    return s


# ---------------------------
# Main pipeline (resumable)
# ---------------------------
def main(
    hf_id: str,
    output_dir: str,
    domain_map: Optional[Dict[str, List[str]]] = None,
    teacher_backend: str = "openai",
    teacher_model: str = "gpt-4o-mini",
    secondary_teacher_model: Optional[str] = None,
    n_solutions: int = 3,
    verify: bool = True,
    sbert_model_name: str = "all-MiniLM-L6-v2",
    sim_threshold: float = 0.90,
    max_problems: Optional[int] = None,
    flatten: bool = True,
    sample_seed: int = 42,
    decontam_eval: Optional[List[str]] = None,
    resume: bool = True,
    checkpoint_interval: int = 20,
) -> None:
    random.seed(sample_seed)
    os.makedirs(output_dir, exist_ok=True)

    if domain_map is None:
        domain_map = {
            "algebra": ["Algebra", "Intermediate Algebra"],
            "geometry": ["Geometry"],
            "statistics": ["Counting & Probability", "Probability", "Combinatorics"],
            "calculus": ["Calculus", "Precalculus"],
        }

    logging.info("Loading dataset: %s", hf_id)
    ds_train = load_dataset(hf_id, split="train")
    try:
        load_dataset(hf_id, split="test")
    except Exception:
        logging.info("No explicit test split for dataset %s", hf_id)

    candidate_fields = [
        f
        for f in ("type", "category", "problem_type", "domain")
        if f in ds_train.column_names
    ]
    if not candidate_fields:
        logging.error(
            "No domain-like field found. Dataset columns: %s", ds_train.column_names
        )
        raise SystemExit(1)
    domain_field = candidate_fields[0]
    logging.info("Using domain field: %s", domain_field)

    eval_ngrams = set()
    if decontam_eval:
        eval_ngrams = compute_eval_ngrams_from_datasets(decontam_eval)
        logging.info("Built decontam ngram set size: %d", len(eval_ngrams))

    # init hf pipeline if needed
    hf_pipe_obj = None
    if teacher_backend == "hf":
        logging.info("Initializing HF generation pipeline model=%s", teacher_model)
        hf_pipe_obj = init_hf_pipeline(teacher_model)

    # init sbert if asked
    sbert_model = None
    if sbert_model_name:
        try:
            logging.info("Loading SBERT model: %s", sbert_model_name)
            sbert_model = init_sbert(sbert_model_name)
        except Exception as e:
            logging.warning(
                "SBERT init failed: %s. Semantic dedupe will be disabled.", e
            )
            sbert_model = None

    teacher_records_file = os.path.join(output_dir, "teacher_records.jsonl")
    existing_teacher_map: Dict[str, List[Dict[str, Any]]] = {}
    if resume and os.path.exists(teacher_records_file):
        logging.info("Resuming from existing teacher_records: %s", teacher_records_file)
        with open(teacher_records_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = normalize_text(rec.get("problem", ""))
                    existing_teacher_map[key] = rec.get("teacher_outputs", [])
                except Exception:
                    continue

    overall_stats: Dict[str, int] = defaultdict(int)

    for vt_domain, cats in domain_map.items():
        logging.info("Domain %s -> categories %s", vt_domain, cats)

        def keep_fn(
            example: Dict[str, Any], cats: List[str] = cats, field: str = domain_field
        ) -> bool:
            val = example.get(field, "")
            if isinstance(val, list):
                return any(c in val for c in cats)
            return any(c in str(val) for c in cats)

        ds_filtered = ds_train.filter(keep_fn)
        logging.info("Found %d candidates for %s", len(ds_filtered), vt_domain)

        problems = []
        for ex in ds_filtered:
            prob_text = ex.get("problem", "") or ex.get("question", "") or ""
            gold_solution = ex.get("solution", "") or ex.get("answer", "") or None
            if not prob_text.strip():
                continue
            problems.append(
                {
                    "problem": normalize_text(prob_text),
                    "gold_solution": gold_solution,
                    "source": hf_id,
                }
            )
        logging.info("Prepared %d normalized problems", len(problems))

        if max_problems:
            problems = random.sample(problems, min(max_problems, len(problems)))
            logging.info("Truncated to max_problems=%d", len(problems))

        domain_out = []
        pbar = tqdm(problems, desc=f"{vt_domain[:6]}", unit="ex")
        for idx, item in enumerate(pbar):
            prob_norm = item["problem"]
            gold = item["gold_solution"]
            key = prob_norm

            if eval_ngrams:
                toks = prob_norm.split()
                contaminated = False
                if len(toks) >= 10:
                    for i in range(len(toks) - 9):
                        if " ".join(toks[i : i + 10]) in eval_ngrams:
                            contaminated = True
                            break
                if contaminated:
                    overall_stats["decontaminated"] += 1
                    continue

            teacher_outputs = existing_teacher_map.get(key)
            if not teacher_outputs:
                teacher_outputs = distill_multiple_solutions(
                    problem=prob_norm,
                    gold_solution=gold,
                    teacher_backend=teacher_backend,
                    teacher_model=teacher_model,
                    n_solutions_target=n_solutions,
                    verify=verify,
                    hf_pipeline_obj=hf_pipe_obj,
                    sbert_model=sbert_model,
                    sim_threshold=sim_threshold,
                    secondary_teacher_model=secondary_teacher_model,
                )
                rec = {"problem": prob_norm, "teacher_outputs": teacher_outputs}
                with open(teacher_records_file, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing_teacher_map[key] = teacher_outputs
                time.sleep(0.05)

            overall_stats["processed"] += 1

            # acceptance policy: keep only verified OR optionally keep unverified but tagged
            accepted = []
            for o in teacher_outputs:
                if o.get("verified", False):
                    accepted.append(o)
            # fallback: if none verified but we have some outputs, accept them into a separate bucket (tagged)
            if not accepted and teacher_outputs:
                overall_stats["accepted_unverified"] += len(teacher_outputs)
                # depending on policy, either skip or include; here we include but mark verified=False
                accepted = teacher_outputs

            if flatten:
                for a in accepted:
                    ans = (
                        a.get("answer")
                        or extract_boxed_answer(a.get("solution", "") or "")
                        or None
                    )
                    recobj = {
                        "problem": prob_norm,
                        "answer": ans,
                        "solution": a.get("solution"),
                        "domain": vt_domain,
                        "source": item.get("source"),
                        "teacher_meta": a.get("meta", {}),
                        "verified": a.get("verified", False),
                        "verify_info": a.get("verify_info", {}),
                    }
                    domain_out.append(recobj)
            else:
                domain_out.append(
                    {
                        "problem": prob_norm,
                        "domain": vt_domain,
                        "source": item.get("source"),
                        "gold_solution": gold,
                        "teacher_outputs": accepted,
                    }
                )

            if (idx + 1) % checkpoint_interval == 0:
                out_file = os.path.join(
                    output_dir,
                    f"spectrum_{vt_domain}_{'flat' if flatten else 'nested'}.jsonl",
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    for entry in domain_out:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                logging.info(
                    "Checkpoint wrote %d domain entries to %s",
                    len(domain_out),
                    out_file,
                )

        out_file = os.path.join(
            output_dir, f"spectrum_{vt_domain}_{'flat' if flatten else 'nested'}.jsonl"
        )
        with open(out_file, "w", encoding="utf-8") as f:
            for entry in domain_out:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(
            "Completed domain %s: wrote %d entries to %s",
            vt_domain,
            len(domain_out),
            out_file,
        )

    logging.info("Done. stats: %s", dict(overall_stats))


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", default="hendrycks/competition_math")
    parser.add_argument("--out", default="data/spectrum_patched")
    parser.add_argument("--teacher-backend", choices=["openai", "hf"], default="openai")
    parser.add_argument("--teacher-model", default="gpt-4o-mini")
    parser.add_argument(
        "--secondary-teacher-model",
        default=None,
        help="Optional second teacher model id for consensus",
    )
    parser.add_argument("--n-solutions", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--sbert-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model for semantic dedupe",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.90,
        help="Semantic similarity threshold for dedupe (0-1); lower is more strict",
    )
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--decontam-eval", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    args = parser.parse_args()

    if args.teacher_backend == "openai" and openai is None:
        logging.error("openai package not installed. pip install openai")
        sys.exit(1)
    if args.teacher_backend == "hf" and hf_pipeline_constructor is None:
        logging.error(
            "transformers pipeline not installed. pip install transformers accelerate"
        )
        sys.exit(1)
    if args.sbert_model and SentenceTransformer is None:
        logging.error(
            "sentence-transformers not installed. pip install sentence-transformers"
        )
        sys.exit(1)
    if args.verify and sp is None:
        logging.warning(
            "sympy not installed — verification will not use SymPy (`--verify` will still run gold/consensus checks)."
        )

    # check OPENAI_API_KEY if openai backend
    if args.teacher_backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logging.warning(
            "OPENAI_API_KEY not found in env. OpenAI calls will fail unless set."
        )

    main(
        hf_id=args.hf_id,
        output_dir=args.out,
        teacher_backend=args.teacher_backend,
        teacher_model=args.teacher_model,
        secondary_teacher_model=args.secondary_teacher_model,
        n_solutions=args.n_solutions,
        verify=args.verify,
        sbert_model_name=args.sbert_model,
        sim_threshold=args.sim_threshold,
        max_problems=args.max_problems,
        flatten=args.flatten,
        sample_seed=args.sample_seed,
        decontam_eval=args.decontam_eval,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
    )
