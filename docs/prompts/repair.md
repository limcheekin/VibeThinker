Context
-------
You have a repository whose spec is `docs/implementation_guide.md`. A prior review (attached below) produced a list of issues: Critical, Major, and Minor. Your job is to implement fixes so the repository **fully complies** with `docs/implementation_guide.md`, is correct, well-tested, and CI-green.

(If you were not given the review as input, start by running the review prompt against the repo, produce the report, then fix items found.)

High-level objective
--------------------
Produce a pull request (branch: `jules/fix-spec-<shortid>`) that fixes all Critical and Major issues, addresses as many Minor issues as reasonably possible, and leaves a clear, actionable TODO list for any remaining low-impact items. The PR must include code, tests, CI updates, documentation, and a `PR_SUMMARY.md` describing changes and verification steps.

Hard constraints (must follow)
-----------------------------
1. Do **not** change the contract defined in `docs/implementation_guide.md` unless there is an explicit, documented contradiction; if you must change the contract, add a single short RFC file `docs/IMPLEMENTATION_DECISIONS.md` that quotes the spec, explains the ambiguity/contradiction, details the chosen design, and gets approved by a reviewer in the PR description.  
2. Target Python **3.11**. Use `pyproject.toml` for packaging.  
3. All production code must include type annotations for exported functions/classes.  
4. Tests: use `pytest` + `pytest-cov`. Unit tests must be deterministic and not rely on real network or filesystem state (mock external IO).  
5. CI: `.github/workflows/ci.yml` must run lint (black/isort), typecheck (mypy), and tests with coverage. CI must fail if coverage for `src/<package_name>` < 90% (unless the spec prescribes different threshold; follow the spec).  
6. No secrets or API keys committed. Use environment variables for runtime secrets and mocks in tests.

Inputs
------
- The repository (root).  
- `docs/implementation_guide.md` (the spec).  
- The prior review report (if available). If no review provided, run the review prompt first and treat its output as the list of issues to fix.

Primary deliverables (what to produce)
-------------------------------------
1. A new branch `jules/fix-spec-<shortid>` with a single cohesive PR that:
   - Fixes Critical and Major issues.
   - Adds or updates tests to cover fixed behaviors (unit + a few integration tests).
   - Updates CI to enforce checks.
   - Updates README and `PR_SUMMARY.md`.
2. `PR_SUMMARY.md` in repo root that lists:
   - Short description of fixes.
   - Files changed.
   - Commands to reproduce tests and lint locally.
   - Coverage percentage and where to find reports.
   - Any remaining TODOs/low-impact items with clear justification (if any).
3. Inline code comments only where necessary to explain non-obvious logic.
4. A clear commit history with small, focused commits (e.g., `fix(core): handle X edge case`, `test(core): add tests for Y`).

Detailed tasks (do all that apply)
---------------------------------
A. **Follow the spec line-by-line**
   - For every requirement in `docs/implementation_guide.md`, ensure there is a corresponding implementation or a documented, justified deviation in `docs/IMPLEMENTATION_DECISIONS.md`.
   - In the PR description, quote the spec paragraph(s) and point to the file/lines implementing them.

B. **Fix functional bugs**
   - Reproduce failing behaviors via new tests (regression tests).
   - Implement fixes with unit tests that demonstrate the bug is resolved.
   - Add property/behavior tests for all previously untested edge cases flagged in the review.

C. **Improve test coverage**
   - Add unit tests for any public function or class identified as missing tests.
   - Add parameterized tests for boundary conditions.
   - Mock external IO with `pytest-mock`/`unittest.mock` and ensure tests don't call external services.
   - Add at least one integration-style test if the project coordinates multiple components.

D. **Type checking & linting**
   - Fix mypy errors. If any mypy ignores are required, add a minimal and justified `# type: ignore[reason]` and document it in `PR_SUMMARY.md`.
   - Ensure `black --check` and `isort --check` pass.
   - Keep ruff/flake8 warnings at zero if possible; document any trade-offs.

E. **CI**
   - Create/update `.github/workflows/ci.yml` to run: lint -> typecheck -> tests with coverage.
   - Cache dependencies for speed.
   - Fail pipeline for coverage below threshold.
   - Add a job step to print a short test summary and coverage percentage.

F. **Error handling & logging**
   - Replace any `print()` used for status with `logging` calls; provide a single module-level logger per module.
   - Add or improve custom exceptions in `src/<package_name>/errors.py` where appropriate.
   - Ensure error messages are clear and tested.

G. **Documentation**
   - Update `README.md` with exact commands:
     - `python -m pip install -e .[dev]` or `poetry install`
     - `pytest -q --cov=src/<package_name> --cov-report=term-missing`
     - `black . --check`, `isort . --check`, `mypy src`
   - Add `PR_SUMMARY.md` that lists verification steps.

H. **PR metadata**
   - PR title: `fix: address review issues from docs/implementation_guide.md`
   - PR body must include:
     - The quoted spec lines changed/implemented.
     - The list of issues fixed (reference the review report item IDs).
     - How to test locally (commands).
     - Any risk/impact notes and remaining small todos.

Quality & acceptance criteria (automated checks + manual)
---------------------------------------------------------
- All tests pass locally: `pytest -q` returns zero failures.
- Coverage for `src/<package_name>` is ≥ 90% (or spec threshold).
- `black --check .`, `isort --check .`, and `mypy src` pass.
- CI workflow passes on the created branch.
- Each Critical item from the review is resolved (no bypasses).
- PR contains `PR_SUMMARY.md` and `docs/IMPLEMENTATION_DECISIONS.md` if any deviations were made.

Reporting back to me (final output)
-----------------------------------
When you finish, in the PR description and in a final reply here, provide:
1. The PR branch name and commit SHA of the head commit.
2. A short summary of what changed (3–6 bullets).
3. The exact commands to reproduce checks locally.
4. Coverage percentage and where to find the HTML report (if generated).
5. Any remaining recommended follow-ups (if any), each with severity and rationale.

Edge-case handling & developer convenience
------------------------------------------
- If fixing a bug requires a breaking API change, do not implement it silently. Instead:
  - Propose the change in `docs/IMPLEMENTATION_DECISIONS.md`.
  - Implement it on a feature branch and mark it as `breaking:` in PR title.
  - Provide migration notes and tests for both old and new behavior if feasible.
- If the repository uses external services (APIs/DB), replace integration points with interfaces/adapters and mock them in tests.

Style & tone
------------
- Be precise and blunt in commit messages and PR description. No fluff.
- When suggesting code changes in comments, show the minimal patch or test that proves the fix.
- Keep changes small and reviewable; do not bundle unrelated refactors into the same PR.

Failure modes
-------------
- If you cannot reproduce a reported issue, add a test demonstrating the attempted reproduction and explain why it's not reproducible. Mark that review item as “unable to reproduce” with steps taken.
- If a requested change would cause the project to diverge from the spec, do not implement it—document the reasoning and propose alternatives.

Example checklist for the PR (to include)
------------------------------------------
- [ ] All Critical issues fixed and tested
- [ ] All Major issues fixed and tested
- [ ] Minor issues resolved or documented
- [ ] `pytest` passes, coverage >= threshold
- [ ] `mypy`, `black`, `isort` pass
- [ ] CI workflow added/updated and passes
- [ ] `PR_SUMMARY.md` added with verification steps
- [ ] `docs/IMPLEMENTATION_DECISIONS.md` added if any deviations

Start now
---------
Create the branch, implement the fixes, run the checks and tests, open the PR, and return the PR link, branch name, and summary as specified.

