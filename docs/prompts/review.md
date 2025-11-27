Your task
---------
Act as a senior Python reviewer. Review the entire repository implementation against the specification in `docs/implementation_guide.md`. Your job is to determine whether the implementation is:

1. Fully compliant with the guide  
2. Correct (functionally and logically)  
3. Complete (all required behaviors implemented)  
4. High-quality (architecture, correctness, tests, packaging, linting)

If any part fails these criteria, you must identify it precisely.

What to review
--------------
1. All modules in `src/` and subpackages  
2. All tests under `tests/`  
3. pyproject.toml, code style configs, and CI workflow files  
4. README (installation, usage, developer docs)  
5. Any CLI or integration layers  
6. Any discrepancies with the contract, rules, workflows, and examples inside `docs/implementation_guide.md`

How to review
-------------
- Compare each requirement from `docs/implementation_guide.md` point-by-point against the written code.
- List missing features, deviations, incorrect behaviors, and mismatches.
- Identify poor assumptions, undocumented behavior, or implicit interpretations not allowed by the spec.
- Flag unclear or brittle logic.
- Validate type hints, error handling, and edge cases.
- Validate that tests cover required scenarios.
- Validate test quality: correctness, clarity, parametrization, mocking, failure cases.
- Validate packaging, project structure, naming, and module layout.
- Validate CI behavior: should lint, check types, run tests, and enforce coverage.
- Validate the README: accuracy, completeness, correctness.

Deliverables
------------
Return a structured review with:

1. **Pass/Fail verdict** (overall compliance)
2. **List of violations**  
   - Each should quote the relevant line/section from `docs/implementation_guide.md`
   - And point to the file/line in the repo that violates it
3. **Missing implementations**  
4. **Incorrect implementations**  
5. **Logical bugs or edge-case failures**
6. **Test coverage gaps**  
   - Identify functions/methods without corresponding tests
7. **Code quality issues**
8. **Security or reliability issues**
9. **Recommended corrections**  
   - Clear, actionable, and specific

Severity levels:
- **Critical** — violates spec; must be fixed
- **Major** — functional but wrong/fragile
- **Minor** — style, cleanup, low-impact polish

Output format
-------------
Return results in this structure:

```

# Review Summary

## Verdict

<pass/fail>

## Critical Issues

* <issue>
  - Spec reference: <quote>
  - Code reference: <file:line>

## Major Issues

* ...

## Minor Issues

* ...

## Missing or Underspecified Parts

* ...

## Test Assessment

* Coverage gaps:
* Incorrect tests:
* Missing edge-case tests:

## Recommendations

* Concrete steps to fix each issue

```

Rules
-----
- Do not rewrite code.
- Do not make assumptions not grounded in the spec.
- Point to exact references for both spec and code when reporting mismatches.
- If compliant, explicitly confirm that each major section of the guide is satisfied.
