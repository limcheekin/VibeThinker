# VibeThinker

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/limcheekin/VibeThinker)

The objective of this project is to reproduce the methodology (SSP, MGPO) from the paper on a smaller scale using the `unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit` model, to see how it compares to the results of the paper’s 1.5B model.

## Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/limcheekin/VibeThinker.git
    cd VibeThinker
    ```

2.  **Install dependencies:**
    This project uses `pyproject.toml` to manage dependencies. Install the project in editable mode with the development dependencies:
    ```bash
    pip install -e .[dev]
    ```

## Running Tests

This project uses `pytest` for testing. To run the tests and see a coverage report:

```bash
pytest --cov=src/vibethinker --cov-report=term-missing
```

## Running Linters

This project uses `black`, `isort`, and `flake8` for linting and code formatting.

-   **Check formatting:**
    ```bash
    black --check .
    isort --check .
    ```

-   **Apply formatting:**
    ```bash
    black .
    isort .
    ```

-   **Run flake8:**
    ```bash
    flake8 src
    ```

## Running the Project

The main entry point for the project is `pipeline.sh`. To run the full training pipeline:

```bash
./pipeline.sh
```
