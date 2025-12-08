#!/usr/bin/env python3
"""
APPS Dataset Preparation for Code Generalization Stage.
Downloads and filters APPS dataset for function-based problems compatible with monitor.py.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm


def is_call_based_problem(sample: Dict[str, Any]) -> bool:
    """
    Determine if a problem is call-based (function execution) vs stdin/stdout.

    Args:
        sample: APPS dataset sample

    Returns:
        True if problem is call-based (has function signature)
    """
    # Check if input_output has fn_name field
    if "input_output" in sample and sample["input_output"]:
        try:
            io_data = json.loads(sample["input_output"])
            if "fn_name" in io_data and io_data["fn_name"]:
                return True
        except (json.JSONDecodeError, TypeError):
            pass

    # Check if starter_code has function definition
    if "starter_code" in sample and sample["starter_code"]:
        if "def " in sample["starter_code"]:
            return True

    return False


def format_test_cases(input_output_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse and format APPS test cases for monitor.py compatibility.

    Args:
        input_output_str: JSON string containing test case data

    Returns:
        List of test cases formatted as {"input": [args], "output": result}
        or None if parsing fails
    """
    try:
        io_data = json.loads(input_output_str)

        # APPS structure: {"inputs": [...], "outputs": [...], "fn_name": "..."}
        if "inputs" not in io_data or "outputs" not in io_data:
            return None

        inputs = io_data["inputs"]
        outputs = io_data["outputs"]

        if len(inputs) != len(outputs):
            return None

        test_cases = []
        for inp, out in zip(inputs, outputs):
            # Parse input - could be JSON list or single value
            try:
                if isinstance(inp, str):
                    parsed_input = json.loads(inp)
                else:
                    parsed_input = inp

                # Ensure input is a list for unpacking in monitor.py
                if not isinstance(parsed_input, list):
                    parsed_input = [parsed_input]

                # Parse output
                if isinstance(out, str):
                    parsed_output = json.loads(out)
                else:
                    parsed_output = out

                test_cases.append({"input": parsed_input, "output": parsed_output})
            except (json.JSONDecodeError, TypeError):
                # Skip malformed test case
                continue

        return test_cases if test_cases else None

    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def prepare_apps_dataset(
    output_dir: str = "data",
    max_train: int = 5000,
    max_val: int = 500,
    difficulty_filter: Optional[List[str]] = None,
) -> None:
    """
    Download and prepare APPS dataset for VibeThinker Code RL.

    Args:
        output_dir: Directory to save processed data
        max_train: Maximum training samples
        max_val: Maximum validation samples
        difficulty_filter: Optional list of difficulties to include
            (e.g., ["introductory", "interview"])
    """
    print("Loading APPS dataset from HuggingFace...")
    dataset = load_dataset("codeparrot/apps", split="train")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_data: List[Dict[str, Any]] = []
    val_data: List[Dict[str, Any]] = []

    print(f"\nProcessing {len(dataset)} problems...")

    for sample in tqdm(dataset):
        # Filter by difficulty if specified
        if difficulty_filter and sample.get("difficulty") not in difficulty_filter:
            continue

        # Filter for call-based problems only
        if not is_call_based_problem(sample):
            continue

        # Parse and format test cases
        test_cases = format_test_cases(sample["input_output"])
        if not test_cases:
            continue

        # Create training example
        example = {
            "problem": sample["question"],
            "test_cases": test_cases,
            "difficulty": sample.get("difficulty", "unknown"),
            "source": "APPS",
        }

        # Split into train/val (90/10 split)
        if len(train_data) < max_train:
            train_data.append(example)
        elif len(val_data) < max_val:
            val_data.append(example)
        else:
            break

    # Save datasets
    train_file = output_path / "apps_train.jsonl"
    val_file = output_path / "apps_val.jsonl"

    print(f"\nSaving {len(train_data)} training examples to {train_file}")
    with open(train_file, "w") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")

    print(f"Saving {len(val_data)} validation examples to {val_file}")
    with open(val_file, "w") as f:
        for example in val_data:
            f.write(json.dumps(example) + "\n")

    # Print statistics
    print("\n" + "=" * 60)
    print("APPS Dataset Preparation Complete")
    print("=" * 60)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    if train_data:
        print("\\nExample problem:")
        print(f"  Problem: {train_data[0]['problem'][:100]}...")
        print(f"  Test cases: {len(train_data[0]['test_cases'])}")
        print(f"  Difficulty: {train_data[0]['difficulty']}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare APPS dataset for VibeThinker Code RL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=5000,
        help="Maximum training samples",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=500,
        help="Maximum validation samples",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        nargs="+",
        choices=["introductory", "interview", "competition"],
        help="Filter by difficulty levels",
    )

    args = parser.parse_args()

    prepare_apps_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        difficulty_filter=args.difficulty,
    )
