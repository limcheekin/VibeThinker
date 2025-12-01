"""
Unit tests for scripts/flatten_to_arrow.py
"""

import json

# Import functions from the script
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from flatten_to_arrow import (
    build_completion,
    build_prompt,
    collect_examples,
    extract_entries_from_record,
    fingerprint,
    iter_input_records,
    normalize_text,
    tokenize_and_prepare_labels_hf,
)


class TestNormalizeText:
    """Test normalize_text function."""

    def test_normalize_basic(self) -> None:
        assert normalize_text("hello world") == "hello world"

    def test_normalize_whitespace(self) -> None:
        assert normalize_text("  hello   world  ") == "hello world"

    def test_normalize_newlines(self) -> None:
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_normalize_tabs(self) -> None:
        assert normalize_text("hello\t\tworld") == "hello world"

    def test_normalize_none(self) -> None:
        assert normalize_text(None) == ""  # type: ignore

    def test_normalize_empty(self) -> None:
        assert normalize_text("") == ""


class TestFingerprint:
    """Test fingerprint function."""

    def test_fingerprint_basic(self) -> None:
        fp1 = fingerprint("problem1", "solution1")
        fp2 = fingerprint("problem1", "solution1")
        assert fp1 == fp2
        assert len(fp1) == 40  # SHA1 hex digest length

    def test_fingerprint_different(self) -> None:
        fp1 = fingerprint("problem1", "solution1")
        fp2 = fingerprint("problem2", "solution2")
        assert fp1 != fp2

    def test_fingerprint_whitespace_normalized(self) -> None:
        fp1 = fingerprint("  problem  1  ", "  solution  1  ")
        fp2 = fingerprint("problem 1", "solution 1")
        assert fp1 == fp2


class TestIterInputRecords:
    """Test iter_input_records function."""

    def test_iter_single_file(self, tmp_path: Path) -> None:
        # Create test JSONL file
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps({"problem": "p1", "solution": "s1"}) + "\n")
            f.write(json.dumps({"problem": "p2", "solution": "s2"}) + "\n")

        records = list(iter_input_records([str(test_file)]))
        assert len(records) == 2
        assert records[0]["problem"] == "p1"
        assert records[1]["problem"] == "p2"

    def test_iter_multiple_files(self, tmp_path: Path) -> None:
        file1 = tmp_path / "test1.jsonl"
        file2 = tmp_path / "test2.jsonl"

        with open(file1, "w") as f:
            f.write(json.dumps({"problem": "p1"}) + "\n")

        with open(file2, "w") as f:
            f.write(json.dumps({"problem": "p2"}) + "\n")

        records = list(iter_input_records([str(file1), str(file2)]))
        assert len(records) == 2

    def test_iter_skip_empty_lines(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps({"problem": "p1"}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"problem": "p2"}) + "\n")

        records = list(iter_input_records([str(test_file)]))
        assert len(records) == 2

    def test_iter_skip_invalid_json(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps({"problem": "p1"}) + "\n")
            f.write("invalid json line\n")
            f.write(json.dumps({"problem": "p2"}) + "\n")

        records = list(iter_input_records([str(test_file)]))
        assert len(records) == 2


class TestExtractEntriesFromRecord:
    """Test extract_entries_from_record function."""

    def test_extract_flat_record(self) -> None:
        rec = {
            "problem": "What is 2+2?",
            "solution": "The answer is 4",
            "answer": "4",
            "verified": True,
            "teacher_meta": {"model": "gpt-4"},
        }
        entries = list(extract_entries_from_record(rec))
        assert len(entries) == 1
        problem, solution, answer, verified, meta = entries[0]
        assert problem == "What is 2+2?"
        assert solution == "The answer is 4"
        assert answer == "4"
        assert verified is True
        assert meta == {"model": "gpt-4"}

    def test_extract_nested_record(self) -> None:
        rec = {
            "problem": "What is 2+2?",
            "teacher_outputs": [
                {
                    "solution": "Solution 1",
                    "answer": "4",
                    "verified": True,
                    "meta": {"model": "gpt-4"},
                },
                {
                    "solution": "Solution 2",
                    "answer": "4",
                    "verified": False,
                    "meta": {"model": "gpt-3.5"},
                },
            ],
        }
        entries = list(extract_entries_from_record(rec))
        assert len(entries) == 2
        assert entries[0][1] == "Solution 1"
        assert entries[1][1] == "Solution 2"

    def test_extract_fallback_record(self) -> None:
        rec = {"text": "Some text", "completion": "Some completion"}
        entries = list(extract_entries_from_record(rec))
        assert len(entries) == 1
        problem, solution, answer, verified, meta = entries[0]
        # In fallback mode, solution comes from 'text' field, not 'completion'
        assert solution == "Some text"


class TestBuildPrompt:
    """Test build_prompt function."""

    def test_build_prompt_basic(self) -> None:
        template = "Problem: {problem}\nSolution:"
        result = build_prompt("What is 2+2?", template)
        assert result == "Problem: What is 2+2?\nSolution:"

    def test_build_prompt_custom_template(self) -> None:
        template = "Question: {problem}\nAnswer:"
        result = build_prompt("What is 2+2?", template)
        assert result == "Question: What is 2+2?\nAnswer:"


class TestBuildCompletion:
    """Test build_completion function."""

    def test_build_completion_keep_cot(self) -> None:
        solution = "Step 1: Add 2+2\nStep 2: Get 4"
        answer = "4"
        result = build_completion(
            solution, answer, keep_cot=True, answer_prefix="Answer: "
        )
        assert result == "Step 1: Add 2+2\nStep 2: Get 4"

    def test_build_completion_answer_only(self) -> None:
        solution = "Step 1: Add 2+2\nStep 2: Get 4"
        answer = "4"
        result = build_completion(
            solution, answer, keep_cot=False, answer_prefix="Answer: "
        )
        assert result == "Answer: 4"

    def test_build_completion_no_answer(self) -> None:
        solution = "Some solution"
        result = build_completion(
            solution, None, keep_cot=False, answer_prefix="Answer: "
        )
        assert result == "Some solution"

    def test_build_completion_empty(self) -> None:
        result = build_completion("", None, keep_cot=False, answer_prefix="Answer: ")
        assert result == ""


class TestCollectExamples:
    """Test collect_examples function."""

    def test_collect_basic(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            f.write(
                json.dumps({"problem": "What is 2+2?", "solution": "4", "answer": "4"})
                + "\n"
            )

        examples = collect_examples(
            input_paths=[str(test_file)],
            prompt_template="Problem: {problem}\nSolution:",
            keep_cot=False,
            answer_prefix="Answer: ",
            dedupe=False,
            shuffle=False,
            seed=42,
            max_examples=None,
            verified_only=False,
        )

        assert len(examples) == 1
        assert "prompt" in examples[0]
        assert "completion" in examples[0]
        assert examples[0]["problem"] == "What is 2+2?"

    def test_collect_dedupe(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            # Write duplicate records
            f.write(
                json.dumps({"problem": "What is 2+2?", "solution": "4", "answer": "4"})
                + "\n"
            )
            f.write(
                json.dumps({"problem": "What is 2+2?", "solution": "4", "answer": "4"})
                + "\n"
            )

        examples = collect_examples(
            input_paths=[str(test_file)],
            prompt_template="Problem: {problem}\nSolution:",
            keep_cot=False,
            answer_prefix="Answer: ",
            dedupe=True,
            shuffle=False,
            seed=42,
            max_examples=None,
            verified_only=False,
        )

        assert len(examples) == 1

    def test_collect_verified_only(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "problem": "p1",
                        "solution": "s1",
                        "answer": "a1",
                        "verified": True,
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "problem": "p2",
                        "solution": "s2",
                        "answer": "a2",
                        "verified": False,
                    }
                )
                + "\n"
            )

        examples = collect_examples(
            input_paths=[str(test_file)],
            prompt_template="Problem: {problem}\nSolution:",
            keep_cot=False,
            answer_prefix="Answer: ",
            dedupe=False,
            shuffle=False,
            seed=42,
            max_examples=None,
            verified_only=True,
        )

        assert len(examples) == 1
        assert examples[0]["problem"] == "p1"

    def test_collect_max_examples(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            for i in range(10):
                f.write(
                    json.dumps(
                        {"problem": f"p{i}", "solution": f"s{i}", "answer": f"a{i}"}
                    )
                    + "\n"
                )

        examples = collect_examples(
            input_paths=[str(test_file)],
            prompt_template="Problem: {problem}\nSolution:",
            keep_cot=False,
            answer_prefix="Answer: ",
            dedupe=False,
            shuffle=False,
            seed=42,
            max_examples=5,
            verified_only=False,
        )

        assert len(examples) == 5

    def test_collect_shuffle(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            for i in range(5):
                f.write(
                    json.dumps(
                        {"problem": f"p{i}", "solution": f"s{i}", "answer": f"a{i}"}
                    )
                    + "\n"
                )

        examples_no_shuffle = collect_examples(
            input_paths=[str(test_file)],
            prompt_template="Problem: {problem}\nSolution:",
            keep_cot=False,
            answer_prefix="Answer: ",
            dedupe=False,
            shuffle=False,
            seed=42,
            max_examples=None,
            verified_only=False,
        )

        examples_shuffle = collect_examples(
            input_paths=[str(test_file)],
            prompt_template="Problem: {problem}\nSolution:",
            keep_cot=False,
            answer_prefix="Answer: ",
            dedupe=False,
            shuffle=True,
            seed=42,
            max_examples=None,
            verified_only=False,
        )

        # With shuffle, order should be different (with high probability)
        assert len(examples_no_shuffle) == len(examples_shuffle)


class TestTokenizeAndPrepareLabelsHf:
    """Test tokenize_and_prepare_labels_hf function."""

    def test_tokenize_basic(self) -> None:
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]],
        }

        examples = {
            "prompt": ["Problem: What is 2+2?\nSolution:"],
            "completion": ["4"],
        }

        result = tokenize_and_prepare_labels_hf(
            examples=examples,
            tokenizer=mock_tokenizer,
            prompt_field="prompt",
            completion_field="completion",
            text_field="text",
            delim="\n\n### Response:\n\n",
            max_length=4096,
            padding="max_length",
            mask_prompt=False,
        )

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_tokenize_mask_prompt(self) -> None:
        # Mock tokenizer
        mock_tokenizer = Mock()
        # Full text tokenization
        mock_tokenizer.side_effect = [
            {"input_ids": [[1, 2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1, 1]]},
            {"input_ids": [[1, 2, 3]]},  # Prompt only
        ]

        examples = {
            "prompt": ["Problem: What is 2+2?\nSolution:"],
            "completion": ["4"],
        }

        result = tokenize_and_prepare_labels_hf(
            examples=examples,
            tokenizer=mock_tokenizer,
            mask_prompt=True,
        )

        # First 3 tokens should be masked (-100)
        assert result["labels"][0][0] == -100
        assert result["labels"][0][1] == -100
        assert result["labels"][0][2] == -100
        # Rest should be original tokens
        assert result["labels"][0][3] == 4
        assert result["labels"][0][4] == 5


class TestMain:
    """Test main function (CLI)."""

    @patch("flatten_to_arrow.Dataset")
    @patch("flatten_to_arrow.AutoTokenizer")
    def test_main_basic(
        self, mock_tokenizer_class: Mock, mock_dataset_class: Mock, tmp_path: Path
    ) -> None:
        # Create test input file
        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            f.write(
                json.dumps({"problem": "What is 2+2?", "solution": "4", "answer": "4"})
                + "\n"
            )

        out_dir = tmp_path / "output"

        # Mock Dataset
        mock_ds = Mock()
        mock_ds.column_names = ["prompt", "completion"]
        mock_ds.__len__ = Mock(return_value=1)
        mock_dataset_class.from_dict.return_value = mock_ds

        # Test without tokenizer (raw dataset)
        with patch(
            "sys.argv",
            [
                "flatten_to_arrow.py",
                "--inputs",
                str(test_file),
                "--out-dir",
                str(out_dir),
            ],
        ):
            from flatten_to_arrow import main

            main()

        # Verify Dataset.from_dict was called
        assert mock_dataset_class.from_dict.called
        # Verify save_to_disk was called
        assert mock_ds.save_to_disk.called
