[MODIFY] test_train_complete.py
Mock unsloth and FastLanguageModel to allow tests to run without the dependency.
Remove pytest.skip and implement proper mocking.

Please activate .venv and ensure the following commands executed successfully for changes you made:
- pytest
- black .
- isort .
- flake8 src scripts
- mypy src scripts
