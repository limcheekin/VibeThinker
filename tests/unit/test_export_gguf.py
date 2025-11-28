import sys
import unittest
from unittest.mock import MagicMock, patch

from vibethinker.export_gguf import benchmark_gguf_inference, export_to_gguf


class TestExportGguf(unittest.TestCase):
    @patch("subprocess.run")
    @patch("os.makedirs")
    @patch("os.path.getsize")
    @patch("os.remove")
    def test_export_to_gguf_llama_cpp(
        self, mock_remove, mock_getsize, mock_makedirs, mock_run
    ):
        mock_getsize.return_value = 1024 * 1024  # 1 MB
        export_to_gguf("mock_model_path", "mock_output_path.gguf")

        # Check that the correct commands were executed
        self.assertEqual(mock_run.call_count, 2)
        mock_run.assert_any_call(
            [
                "python",
                "llama.cpp/convert-hf-to-gguf.py",
                "mock_model_path",
                "--outfile",
                "mock_output_path-fp16.gguf",
                "--outtype",
                "f16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        mock_run.assert_any_call(
            [
                "llama.cpp/quantize",
                "mock_output_path-fp16.gguf",
                "mock_output_path.gguf",
                "q4_k_m",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        mock_remove.assert_called_once_with("mock_output_path-fp16.gguf")

    @patch("os.makedirs")
    @patch("os.path.getsize")
    def test_export_to_gguf_optimum(self, mock_getsize, mock_makedirs):
        mock_optimum = MagicMock()
        # Mock the module in sys.modules to avoid ImportError
        with patch.dict(
            "sys.modules",
            {
                "optimum": mock_optimum,
                "optimum.exporters.ggml": mock_optimum.exporters.ggml,
            },
        ):
            mock_getsize.return_value = 1024 * 1024  # 1 MB
            export_to_gguf(
                "mock_model_path", "mock_output_path.gguf", use_llama_cpp=False
            )
            mock_optimum.exporters.ggml.main.assert_called_once()

    def test_benchmark_gguf_inference(self):
        mock_llama_cpp = MagicMock()
        mock_instance = MagicMock()
        mock_instance.return_value = {"choices": [{"text": "This is a test sentence."}]}
        mock_llama_cpp.Llama.return_value = mock_instance

        # Mock the module in sys.modules to avoid ImportError
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            with patch("time.time", side_effect=[0.0, 1.0]):
                benchmark_gguf_inference("mock_gguf_path")

        mock_llama_cpp.Llama.assert_called_once_with(
            model_path="mock_gguf_path", n_ctx=2048, n_threads=8
        )
        # Check that the model was called twice (once for warmup, once for benchmark)
        self.assertEqual(mock_instance.call_count, 2)


if __name__ == "__main__":
    unittest.main()
