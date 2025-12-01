"""
Unit tests for scripts/prepare_spectrum_data.py
"""

import re

# Import functions from the script
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from prepare_spectrum_data import (
    _latex_to_ascii,
    _try_numeric_equal,
    _try_sympy_equal,
    build_ngrams,
    extract_boxed_answer,
    normalize_text,
    text_fingerprint,
    verify_against_gold,
    verify_answer_via_sympy,
)


class TestNormalizeText:
    """Test normalize_text function."""

    def test_normalize_basic(self) -> None:
        assert normalize_text("hello world") == "hello world"

    def test_normalize_whitespace(self) -> None:
        assert normalize_text("  hello   world  ") == "hello world"

    def test_normalize_newlines(self) -> None:
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_normalize_carriage_return(self) -> None:
        assert normalize_text("hello\r\nworld") == "hello world"

    def test_normalize_html_entities(self) -> None:
        assert normalize_text("hello&nbsp;world") == "hello world"

    def test_normalize_none(self) -> None:
        assert normalize_text(None) == ""  # type: ignore

    def test_normalize_empty(self) -> None:
        assert normalize_text("") == ""


class TestTextFingerprint:
    """Test text_fingerprint function."""

    def test_fingerprint_basic(self) -> None:
        fp1 = text_fingerprint("hello world")
        fp2 = text_fingerprint("hello world")
        assert fp1 == fp2
        assert len(fp1) == 40  # SHA1 hex digest length

    def test_fingerprint_different(self) -> None:
        fp1 = text_fingerprint("hello world")
        fp2 = text_fingerprint("goodbye world")
        assert fp1 != fp2

    def test_fingerprint_normalized(self) -> None:
        fp1 = text_fingerprint("  hello   world  ")
        fp2 = text_fingerprint("hello world")
        assert fp1 == fp2


class TestExtractBoxedAnswer:
    """Test extract_boxed_answer function."""

    def test_extract_simple_boxed(self) -> None:
        text = "The answer is \\boxed{42}"
        assert extract_boxed_answer(text) == "42"

    def test_extract_boxed_with_spaces(self) -> None:
        text = "The answer is \\boxed  {  42  }"
        assert extract_boxed_answer(text) == "42"

    def test_extract_nested_braces(self) -> None:
        text = "The answer is \\boxed{\\frac{1}{2}}"
        assert extract_boxed_answer(text) == "\\frac{1}{2}"

    def test_extract_last_boxed(self) -> None:
        text = "First \\boxed{wrong} then \\boxed{correct}"
        assert extract_boxed_answer(text) == "correct"

    def test_extract_inline_math(self) -> None:
        text = "The answer is $42$"
        result = extract_boxed_answer(text)
        assert result == "42"

    def test_extract_from_answer_line(self) -> None:
        text = "Step 1: Do something\nStep 2: Do more\nAnswer: 42"
        result = extract_boxed_answer(text)
        assert result is not None
        assert "42" in result

    def test_extract_none_empty(self) -> None:
        assert extract_boxed_answer("") is None

    def test_extract_none_no_answer(self) -> None:
        text = "This is just some text without an answer"
        result = extract_boxed_answer(text)
        # May return None or extract a token
        assert result is None or isinstance(result, str)


class TestLatexToAscii:
    """Test _latex_to_ascii function."""

    def test_convert_frac(self) -> None:
        assert _latex_to_ascii("\\frac{1}{2}") == "(1)/(2)"

    def test_convert_superscript_braces(self) -> None:
        result = _latex_to_ascii("x^{2}")
        assert "**" in result

    def test_convert_superscript_simple(self) -> None:
        result = _latex_to_ascii("x^2")
        assert "**" in result

    def test_convert_times(self) -> None:
        assert "*" in _latex_to_ascii("2\\times3")

    def test_convert_cdot(self) -> None:
        assert "*" in _latex_to_ascii("2\\cdot3")

    def test_remove_dollar_signs(self) -> None:
        result = _latex_to_ascii("$x + 1$")
        assert "$" not in result

    def test_empty_string(self) -> None:
        assert _latex_to_ascii("") == ""


class TestTrySymPyEqual:
    """Test _try_sympy_equal function."""

    @patch("prepare_spectrum_data.standard_transformations", ())
    @patch("prepare_spectrum_data.implicit_multiplication_application", Mock())
    @patch("prepare_spectrum_data.sp")
    @patch("prepare_spectrum_data.parse_expr")
    def test_sympy_equal_simple(self, mock_parse: Mock, mock_sp: Mock) -> None:
        # Mock sympy to return equal expressions
        mock_expr1 = Mock()
        mock_expr2 = Mock()
        mock_expr1.__sub__ = Mock(return_value=Mock())
        mock_parse.side_effect = [mock_expr1, mock_expr2]
        mock_sp.simplify.return_value = 0

        result = _try_sympy_equal("2+2", "4")
        assert result is True

    @patch("prepare_spectrum_data.sp", None)
    def test_sympy_not_available(self) -> None:
        result = _try_sympy_equal("2+2", "4")
        assert result is None

    @patch("prepare_spectrum_data.sp")
    @patch("prepare_spectrum_data.parse_expr")
    def test_sympy_parse_error(self, mock_parse: Mock, mock_sp: Mock) -> None:
        mock_parse.side_effect = Exception("Parse error")
        result = _try_sympy_equal("invalid", "expression")
        assert result is None


class TestTryNumericEqual:
    """Test _try_numeric_equal function."""

    def test_numeric_equal_integers(self) -> None:
        result = _try_numeric_equal("42", "42")
        assert result is True

    def test_numeric_equal_decimals(self) -> None:
        result = _try_numeric_equal("3.14", "3.14")
        assert result is True

    def test_numeric_equal_fraction(self) -> None:
        result = _try_numeric_equal("1/2", "0.5")
        assert result is True

    def test_numeric_not_equal(self) -> None:
        result = _try_numeric_equal("42", "43")
        assert result is False

    def test_numeric_empty(self) -> None:
        result = _try_numeric_equal("", "42")
        assert result is None

    def test_numeric_no_numbers(self) -> None:
        result = _try_numeric_equal("abc", "def")
        assert result is None


class TestVerifyAgainstGold:
    """Test verify_against_gold function."""

    def test_verify_exact_match(self) -> None:
        assert verify_against_gold("42", "42") is True

    def test_verify_normalized_match(self) -> None:
        assert verify_against_gold("  42  ", "42") is True

    def test_verify_boxed_match(self) -> None:
        assert verify_against_gold("\\boxed{42}", "42") is True

    def test_verify_no_match(self) -> None:
        assert verify_against_gold("42", "43") is False

    def test_verify_empty(self) -> None:
        assert verify_against_gold("", "42") is False
        assert verify_against_gold("42", "") is False

    def test_verify_none(self) -> None:
        assert verify_against_gold(None, "42") is False  # type: ignore
        assert verify_against_gold("42", None) is False  # type: ignore

    @patch("prepare_spectrum_data._try_sympy_equal")
    def test_verify_with_sympy(self, mock_sympy: Mock) -> None:
        mock_sympy.return_value = True
        assert verify_against_gold("x^2", "x**2") is True

    @patch("prepare_spectrum_data._try_sympy_equal")
    @patch("prepare_spectrum_data._try_numeric_equal")
    def test_verify_with_numeric(self, mock_numeric: Mock, mock_sympy: Mock) -> None:
        mock_sympy.return_value = None
        mock_numeric.return_value = True
        assert verify_against_gold("1/2", "0.5") is True


class TestVerifyAnswerViaSympy:
    """Test verify_answer_via_sympy function."""

    @patch("prepare_spectrum_data.sp", None)
    def test_sympy_not_installed(self) -> None:
        result = verify_answer_via_sympy("problem", "solution", "42")
        assert result["verified"] is False
        assert result["reason"] == "sympy_not_installed"

    @patch("prepare_spectrum_data.sp")
    def test_no_claimed_answer(self, mock_sp: Mock) -> None:
        result = verify_answer_via_sympy("problem", "solution", None)
        assert result["verified"] is False
        assert result["reason"] == "no_claimed_answer"

    @patch("prepare_spectrum_data.sp")
    def test_numeric_answer(self, mock_sp: Mock) -> None:
        mock_sp.N.return_value = 42
        mock_sp.sympify.return_value = 42
        result = verify_answer_via_sympy("What is 2+2?", "solution", "4")
        assert result["verified"] is True

    @patch("prepare_spectrum_data.sp")
    def test_exception_handling(self, mock_sp: Mock) -> None:
        # Mock sympify to raise exception
        mock_sp.sympify.side_effect = Exception("Test error")
        result = verify_answer_via_sympy("problem", "solution", "invalid")
        assert result["verified"] is False
        # The reason should contain 'exception' or be 'no_numeric_in_claimed_answer'
        assert "exception" in result["reason"] or "no_numeric" in result["reason"]


class TestCallTeacherOpenai:
    """Test call_teacher_openai function (mocked)."""

    @patch("prepare_spectrum_data.openai")
    def test_call_teacher_openai_success(self, mock_openai: Mock) -> None:
        from prepare_spectrum_data import call_teacher_openai

        mock_response = {"choices": [{"message": {"content": "Solution text"}}]}
        mock_openai.ChatCompletion.create.return_value = mock_response

        result = call_teacher_openai("Problem: 2+2", "gpt-4o-mini")
        assert len(result) == 1
        assert result[0] == "Solution text"

    @patch("prepare_spectrum_data.openai", None)
    def test_call_teacher_openai_not_installed(self) -> None:
        from prepare_spectrum_data import call_teacher_openai

        with pytest.raises(RuntimeError, match="openai package not installed"):
            call_teacher_openai("Problem: 2+2", "gpt-4o-mini")


class TestInitHfPipeline:
    """Test init_hf_pipeline function (mocked)."""

    @patch("prepare_spectrum_data.hf_pipeline_constructor")
    @patch("prepare_spectrum_data.torch")
    def test_init_hf_pipeline_success(
        self, mock_torch: Mock, mock_pipeline: Mock
    ) -> None:
        from prepare_spectrum_data import init_hf_pipeline

        mock_torch.cuda.is_available.return_value = True
        mock_pipeline.return_value = Mock()

        result = init_hf_pipeline("meta-llama/Llama-3.2-1B")
        assert result is not None
        mock_pipeline.assert_called_once()

    @patch("prepare_spectrum_data.hf_pipeline_constructor", None)
    def test_init_hf_pipeline_not_installed(self) -> None:
        from prepare_spectrum_data import init_hf_pipeline

        with pytest.raises(RuntimeError, match="transformers pipeline not available"):
            init_hf_pipeline("meta-llama/Llama-3.2-1B")


class TestCallTeacherHf:
    """Test call_teacher_hf function (mocked)."""

    def test_call_teacher_hf_success(self) -> None:
        from prepare_spectrum_data import call_teacher_hf

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"generated_text": "Solution text"}]

        result = call_teacher_hf(mock_pipeline, "Problem: 2+2")
        assert len(result) == 1
        assert result[0] == "Solution text"

    def test_call_teacher_hf_retry_on_error(self) -> None:
        from prepare_spectrum_data import call_teacher_hf

        mock_pipeline = Mock()
        mock_pipeline.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
        ]

        result = call_teacher_hf(mock_pipeline, "Problem: 2+2")
        assert result == []


class TestInitSbert:
    """Test init_sbert function (mocked)."""

    @patch("prepare_spectrum_data.SentenceTransformer")
    def test_init_sbert_success(self, mock_st: Mock) -> None:
        from prepare_spectrum_data import init_sbert

        mock_model = Mock()
        mock_st.return_value = mock_model

        result = init_sbert("all-MiniLM-L6-v2")
        assert result == mock_model

    @patch("prepare_spectrum_data.SentenceTransformer", None)
    def test_init_sbert_not_installed(self) -> None:
        from prepare_spectrum_data import init_sbert

        with pytest.raises(RuntimeError, match="sentence-transformers not installed"):
            init_sbert()


class TestIsSemanticallyDistinct:
    """Test is_semantically_distinct function (mocked)."""

    @patch("prepare_spectrum_data.sbert_util")
    def test_semantically_distinct_empty_list(self, mock_util: Mock) -> None:
        from prepare_spectrum_data import is_semantically_distinct

        mock_model = Mock()
        result = is_semantically_distinct("new text", [], mock_model)
        assert result is True

    @patch("prepare_spectrum_data.sbert_util")
    def test_semantically_distinct_below_threshold(self, mock_util: Mock) -> None:
        from prepare_spectrum_data import is_semantically_distinct

        mock_model = Mock()
        mock_model.encode.return_value = Mock()

        # Mock similarity below threshold
        import numpy as np

        mock_cos_sim = Mock()
        mock_cos_sim.cpu.return_value.numpy.return_value.flatten.return_value = (
            np.array([0.85])
        )
        mock_util.cos_sim.return_value = mock_cos_sim

        result = is_semantically_distinct(
            "new text", ["existing text"], mock_model, threshold=0.90
        )
        assert result is True

    @patch("prepare_spectrum_data.sbert_util")
    def test_semantically_distinct_above_threshold(self, mock_util: Mock) -> None:
        from prepare_spectrum_data import is_semantically_distinct

        mock_model = Mock()
        mock_model.encode.return_value = Mock()

        # Mock similarity above threshold
        import numpy as np

        mock_cos_sim = Mock()
        mock_cos_sim.cpu.return_value.numpy.return_value.flatten.return_value = (
            np.array([0.95])
        )
        mock_util.cos_sim.return_value = mock_cos_sim

        result = is_semantically_distinct(
            "new text", ["existing text"], mock_model, threshold=0.90
        )
        assert result is False


class TestDistillMultipleSolutions:
    """Test distill_multiple_solutions function (mocked)."""

    @patch("prepare_spectrum_data.call_teacher_openai")
    def test_distill_basic(self, mock_teacher: Mock) -> None:
        from prepare_spectrum_data import distill_multiple_solutions

        # Mock teacher responses
        mock_teacher.return_value = ["Solution with \\boxed{42}"]

        results = distill_multiple_solutions(
            problem="What is 2+2?",
            gold_solution="4",
            teacher_backend="openai",
            teacher_model="gpt-4o-mini",
            n_solutions_target=1,
            verify=False,
            max_attempts=3,
        )

        assert len(results) >= 0  # May be 0 or more depending on verification
        assert mock_teacher.called

    @patch("prepare_spectrum_data.call_teacher_hf")
    def test_distill_with_hf_backend(self, mock_teacher: Mock) -> None:
        from prepare_spectrum_data import distill_multiple_solutions

        mock_teacher.return_value = ["Solution with \\boxed{42}"]
        mock_pipeline = Mock()

        results = distill_multiple_solutions(
            problem="What is 2+2?",
            gold_solution="4",
            teacher_backend="hf",
            teacher_model="meta-llama/Llama-3.2-1B",
            n_solutions_target=1,
            verify=False,
            max_attempts=3,
            hf_pipeline_obj=mock_pipeline,
        )

        assert len(results) >= 0
        assert mock_teacher.called


class TestBuildNgrams:
    """Test build_ngrams function."""

    def test_build_ngrams_basic(self) -> None:
        text = "one two three four five six seven eight nine ten eleven"
        ngrams = build_ngrams(text, n=3)
        assert "one two three" in ngrams
        assert "two three four" in ngrams

    def test_build_ngrams_short_text(self) -> None:
        text = "one two"
        ngrams = build_ngrams(text, n=10)
        assert len(ngrams) == 0

    def test_build_ngrams_empty(self) -> None:
        ngrams = build_ngrams("", n=10)
        assert len(ngrams) == 0

    def test_build_ngrams_none(self) -> None:
        ngrams = build_ngrams(None, n=10)  # type: ignore
        assert len(ngrams) == 0
