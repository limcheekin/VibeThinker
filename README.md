# VibeThinker

This repository contains a Python implementation of the VibeThinker model, based on the provided implementation guide.

## Development Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e .[dev]
    ```

## Running Linters

To check for code formatting and style, run the following commands:

```bash
black --check .
isort --check .
flake8 src tests
mypy src
```

## Running Tests

To run the unit tests and see a coverage report, use the following command:

```bash
pytest --cov=src/vibethinker --cov-report=term-missing
```
