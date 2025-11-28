"""
Export VibeThinker model to GGUF format for efficient inference
on edge devices and CPU-based deployment.
"""

import os
import subprocess


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
    use_llama_cpp: bool = True,
) -> None:
    """
    Export model to GGUF format.

    Args:
        model_path: Path to HuggingFace model
        output_path: Output path for GGUF file
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16)
        use_llama_cpp: Use llama.cpp for conversion
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if use_llama_cpp:
        if not os.path.exists("llama.cpp"):
            print("ERROR: llama.cpp not found in current directory.")
            print("Please run 'scripts/setup_llama_cpp.sh' or clone it manually:")
            print("  git clone https://github.com/ggerganov/llama.cpp")
            print("  cd llama.cpp && make")
            raise FileNotFoundError("llama.cpp directory not found")

        print("Using llama.cpp for GGUF conversion...")
        temp_fp16 = output_path.replace(".gguf", "-fp16.gguf")
        print(f"Step 1/2: Converting {model_path} to FP16 GGUF...")
        convert_cmd = [
            "python",
            "llama.cpp/convert-hf-to-gguf.py",
            model_path,
            "--outfile",
            temp_fp16,
            "--outtype",
            "f16",
        ]
        try:
            subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
            print(f"✓ FP16 conversion complete: {temp_fp16}")
        except subprocess.CalledProcessError as e:
            print(f"Error during FP16 conversion: {e.stderr}")
            raise
        if quantization != "f16":
            print(f"Step 2/2: Quantizing to {quantization}...")
            quantize_cmd = ["llama.cpp/quantize", temp_fp16, output_path, quantization]
            try:
                subprocess.run(quantize_cmd, check=True, capture_output=True, text=True)
                print(f"✓ Quantization complete: {output_path}")
                os.remove(temp_fp16)
            except subprocess.CalledProcessError as e:
                print(f"Error during quantization: {e.stderr}")
                raise
        else:
            os.rename(temp_fp16, output_path)
            print(f"✓ FP16 export complete: {output_path}")
    else:
        print("Using HuggingFace optimum for GGUF conversion...")
        try:
            from optimum.exporters.ggml import main as ggml_export

            ggml_export(
                [
                    "--model",
                    model_path,
                    "--output",
                    output_path,
                    "--quantize",
                    quantization,
                ]
            )
            print(f"✓ GGUF export complete: {output_path}")
        except ImportError:
            print("ERROR: optimum not installed. Install with: pip install optimum")
            raise
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("\nExport Summary:")
    print(f"  Output file: {output_path}")
    print(f"  Size: {output_size_mb:.2f} MB")
    print(f"  Quantization: {quantization}")


def benchmark_gguf_inference(gguf_path: str, prompt: str = "Solve: 2x + 3 = 7") -> None:
    """Benchmark GGUF model inference speed."""
    import time

    print(f"\nBenchmarking GGUF model: {gguf_path}")
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=8)
        _ = llm(prompt, max_tokens=10)
        start = time.time()
        output = llm(prompt, max_tokens=256, temperature=0.7)
        elapsed = time.time() - start
        tokens_generated = len(output["choices"][0]["text"].split())
        tokens_per_sec = tokens_generated / elapsed
        print(f"✓ Generation completed in {elapsed:.2f}s")
        print(f"  Tokens generated: {tokens_generated}")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
        print(f"  Output: {output['choices'][0]['text'][:100]}...")
    except ImportError:
        print("ERROR: llama-cpp-python not installed.")
        print("Install with: pip install llama-cpp-python")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export VibeThinker to GGUF")
    parser.add_argument("--model-path", required=True, help="Path to HuggingFace model")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Quantization type",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark after export"
    )
    args = parser.parse_args()
    export_to_gguf(
        model_path=args.model_path,
        output_path=args.output,
        quantization=args.quantization,
    )
    if args.benchmark:
        benchmark_gguf_inference(args.output)
    print("\n✓ Export complete!")
    print("\nTo use this model:")
    print(f"  llama-cli -m {args.output} -p 'Solve: 2x + 3 = 7'")
