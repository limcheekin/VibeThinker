Context
-------
You are given a repository that contains a file at `docs/implementation_guide.md` (path relative to repo root). That document is the spec. Implement a complete, production-quality Python project that follows that guide **exactly**. If the guide conflicts with these instructions, prefer the guide — but when the guide is ambiguous or missing details, make pragmatic, industry-standard choices described below.

Primary goals
-------------
1. Implement the project features described in `docs/implementation_guide.md`.
2. Produce clean, well-documented, type-annotated Python code that follows modern best practices.
3. Provide thorough test coverage using `pytest` and supporting tools so the core code is well-tested and reliable.
4. Add CI config so the repo runs linting, tests, and coverage checks automatically.
5. Deliver a clear README and a short developer checklist so reviewers can validate correctness quickly.

Environment & tooling (defaults — follow guide if it specifies otherwise)
--------------------------------------------------------------------------
- Target Python: **3.11** (use pyproject.toml and modern packaging)
- Packaging: use **pyproject.toml** (poetry or setuptools via PEP 621 — choose the one that best fits the repo style).
- Formatting: **black** (autoformat), **isort** for imports.
- Static analysis: **mypy** (strict-ish), **ruff** or **flake8** (choose one; configure reasonable rules).
- Testing: **pytest** (with pytest-cov). Use fixtures, parametrization, and clear Arrange/Act/Assert structure.
- CI: **GitHub Actions** workflow that runs lint, mypy, tests with coverage, and fails if coverage < 90% (unless the guide specifies a different threshold).
- Test doubles: use **pytest-mock** or **unittest.mock** as appropriate.
- Docstrings: Google style or NumPy style (pick one and be consistent) and include short module-level docs.
- README: show how to install dev dependencies, run tests, run linters, and run the project.

Coding conventions & quality
----------------------------
- Use explicit type hints everywhere public functions/methods accept or return values.
- Small functions (single responsibility). Keep modules < ~400 lines if possible.
- Avoid global mutable state; prefer dependency injection.
- Raise custom exceptions where meaningful; provide clear error messages.
- Write unit tests for happy path, edge cases, and expected failures.
- For I/O or network interactions, wrap them behind interfaces that are easy to mock.
- Use logging (no print statements) and allow log level configuration.

Repository layout (suggested)
-----------------------------
```

/ (repo root)
├─ pyproject.toml
├─ README.md
├─ docs/
│  └─ implementation_guide.md  # source spec (read & follow)
├─ src/
│  └─ <package_name>/
│     ├─ **init**.py
│     ├─ main.py            # CLI entrypoint or top-level orchestration
│     ├─ core.py            # main domain logic
│     ├─ api.py             # any external integrations (wrap in interfaces)
│     └─ errors.py
├─ tests/
│  ├─ unit/
│  └─ integration/
├─ .github/
│  └─ workflows/ci.yml
└─ tox.ini or .github/ or .pre-commit-config.yaml (if using pre-commit)

```
Adjust names to match the guide.

Tests & coverage expectations
-----------------------------
- Write unit tests that cover:
  - All public functions and methods in `src/<package_name>`
  - Edge cases and expected error paths
  - Input validation
  - Any branching logic (aim for meaningful branch coverage)
- Write a few integration-style tests if the implementation involves multiple components interacting (but mock external services).
- Include tests verifying CLI behavior if the project has a CLI.
- Configure `pytest-cov` and set coverage threshold in CI. Aim for **>= 90%** coverage for core modules; if the guide sets a different target follow that.
- Provide example test commands:
  - `python -m pip install -e .[dev]`
  - `pytest -q --cov=src/<package_name> --cov-report=term-missing`

CI (GitHub Actions)
-------------------
- Single workflow `.github/workflows/ci.yml` with jobs:
  1. `lint` — runs black (check), isort (check), ruff/flake8.
  2. `typecheck` — runs mypy.
  3. `test` — installs dev deps, runs pytest with coverage; fail if coverage < threshold.
- Cache pip/poetry dependencies to speed up runs.
- Comment on PRs with test summary and coverage percentage (optional but helpful).

Deliverables (explicit)
-----------------------
For the PR, include:
1. Implemented code under `src/<package_name>` that matches the guide.
2. `pyproject.toml` (and lockfile if using poetry) or equivalent packaging files.
3. `tests/` with `unit/` and `integration/` tests (pytest).
4. `README.md` with: summary, install, run, test, and lint instructions.
5. `.github/workflows/ci.yml` configured as above.
6. `pre-commit` config or docs explaining local dev checks (optional but preferred).
7. A short **PR description** that lists:
   - Files added/changed,
   - How the implementation follows `docs/implementation_guide.md`,
   - How to run the tests locally,
   - Any design decisions or deviations from the guide (explicitly justify).

Acceptance criteria & verification steps
---------------------------------------
The PR should satisfy these checks before requesting review:
- `pip install -e .[dev]` (or poetry install) completes without error.
- `black --check .` passes.
- `isort --check .` passes.
- `mypy src` returns no errors (allow a small number of tolerable ignores if justified).
- `pytest` passes all tests.
- CI run (GitHub Actions) completes successfully.
- Coverage for `src/<package_name>` meets the threshold declared in CI.

Edge cases & decision policy
----------------------------
- If the `implementation_guide.md` contradicts itself or omits crucial implementation details, choose the most conservative design that preserves correctness and testability. Document that choice in the PR.
- If external API keys or secrets are required, mock them in tests and use environment variables for runtime; never commit secrets.

Helpful hints for implementation
--------------------------------
- Start by writing tests for the core behaviors described in `docs/implementation_guide.md` (TDD-first).
- Implement minimal passing code, then iterate to improve design and test coverage.
- Keep each commit focused and small. Use clear commit messages like: `feat(core): implement X behavior` or `test(core): add tests for X edge cases`.

User / Repo-specific instructions
---------------------------------
- **Open and strictly follow** `docs/implementation_guide.md`. Quote the relevant spec paragraph(s) in your PR where you implement those behaviors.
- Name the package and modules sensibly based on the guide (do **not** invent unrelated names).
- Do not change any external API or spec behavior unless the guide permits it — in which case explain why.

Output format for your response (what I expect you to return)
-------------------------------------------------------------
When you finish, post a PR-style summary in the repo root `PR_SUMMARY.md` that contains:
- Short description of what's implemented.
- Commands to run tests & linters.
- Coverage percentage and where to find reports.
- Any outstanding todos / known issues.

Runbook for reviewers
---------------------
- Run `pip install -e .[dev]`.
- Run `black . --check`, `isort . --check`, `mypy src`, then `pytest -q`.
- Read `README.md` and `PR_SUMMARY.md` and verify the behavior described is present.
```

## Notes

* Do not ask the user configuration questions; pick sensible defaults as stated.
* Keep changes small and well-documented so reviewers can validate quickly.
* If anything in `docs/implementation_guide.md` conflicts materially with these defaults, follow the guide and explicitly explain the discrepancy in the PR.

