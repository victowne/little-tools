# Agent Guidelines for the Stock Valuation Project

These instructions apply to AI coding agents and contributors working in this
repository. They are intended to protect financial correctness, preserve the
project's unified valuation architecture, and keep changes reviewable.

## Priorities

Use this order when trade-offs arise:

1. Financial correctness and explicit missing-data semantics.
2. Preservation of validated valuation behavior.
3. Clear, maintainable code and focused changes.
4. Deterministic tests and reproducible verification.
5. Performance improvements supported by an observed need.

Do not optimize, parallelize, add dependencies, or redesign architecture merely
to satisfy a generic best practice. Prefer the simplest approach that is correct
for the project's small financial-statement datasets.

## Project Architecture

- Keep data acquisition, financial normalization, pure calculations, and
  Streamlit presentation separate where practical.
- Keep valuation calculations independent of Streamlit and network access.
- Reuse the existing multi-stage DCF calculation chain instead of duplicating
  formulas in UI or research modules.
- Use one production DCF architecture for all companies. Represent company
  differences through researched assumptions, evidence, confidence, and model
  limitations rather than ticker-specific valuation engines.
- Avoid broad restructuring or unrelated cleanup while implementing a focused
  request.
- Preserve public behavior unless the task explicitly changes it. Report
  intentional behavior changes.

## Financial Data Semantics

- A genuine reported zero must remain `0.0`.
- Missing fields, `NaN` values, failed lookups, and ambiguous matches must remain
  unavailable, normally represented by `None` or a structured unavailable
  result. Never use zero as a generic missing-data sentinel.
- Financial statement matching must be conservative and use explicit aliases.
  A missing value is preferable to a confidently wrong line item.
- Validated TTM calculations must use four distinct consecutive fiscal quarters.
  Do not silently skip a missing recent quarter and substitute an older one.
- Keep assumptions separate from observed data. When a fallback assumption is
  used, make it visible in code, diagnostics, or result metadata.
- Do not use market price to tune Company Profile assumptions.
- Preserve units explicitly, especially for percentages, currency values, and
  share counts.

## Python Style

- Follow PEP 8 and use four spaces for indentation.
- Use descriptive `snake_case` names for functions and variables,
  `PascalCase` for classes, and `UPPER_CASE` for constants.
- Prefer lines of approximately 88 characters, but allow longer URLs, user-facing
  text, type declarations, or expressions when splitting would reduce clarity.
- Avoid wildcard imports, mutable default arguments, bare `except` clauses,
  commented-out code, debug prints, and committed breakpoints.
- Use dataclasses for focused data containers when they clarify the model.
- Prefer composition and small, single-purpose functions. Parameter count is a
  design signal, not an absolute limit.
- Do not add comments that merely repeat the code. Document financial reasoning,
  units, non-obvious assumptions, and important compatibility behavior.

## Documentation and Type Hints

- Add type hints to new or materially changed production functions.
- Public financial calculation APIs and important result structures must have
  clear docstrings.
- Complex calculations should document inputs, outputs, units, assumptions, and
  meaningful exceptions. Include an example only when it improves understanding.
- Simple private helpers, Streamlit rendering helpers, and self-explanatory test
  functions do not require ceremonial docstrings.
- Prefer precise types and `T | None` for nullable values. Use `Any` only at
  genuinely dynamic third-party or framework boundaries, and contain it locally.
- Improve typing incrementally. Do not perform repository-wide annotation changes
  as part of an unrelated task.

## DataFrame and Serialization Policy

- Pandas is an accepted project dependency because yfinance returns pandas
  objects and Streamlit and Plotly integrate with them directly.
- Do not migrate existing pandas code to Polars without a measured performance or
  scale requirement. Polars may be introduced for future large, batch-oriented
  datasets when it provides a clear benefit.
- The standard-library `json` module is appropriate for normal project use.
  Introduce `orjson` only when profiling demonstrates a meaningful need.
- Avoid unnecessary conversions between dataframe libraries.

## Errors and Logging

- Never silently swallow an exception.
- Catch specific exceptions when their types are stable and known.
- At external-provider or top-level UI boundaries, a final broad exception guard
  is acceptable when it prevents the entire application from failing. It must
  record useful diagnostics and return a clear unavailable state.
- Do not expose secrets, API keys, sensitive URLs, or personal data in logs.
- Use logging for errors and diagnostics. Normal command-line output intended for
  users or machine consumption may use stdout.
- Use context managers for resources that require deterministic cleanup.

## Dependencies and Tooling

- Minimize dependencies. Add one only when it materially improves correctness,
  maintainability, or required functionality.
- Keep all runtime and development dependencies documented. Existing
  `requirements.txt` and `requirements-dev.txt` remain valid until the project
  deliberately adopts `pyproject.toml`.
- `pyproject.toml` and a lockfile are recommended for future dependency
  modernization, but introducing them must be a focused task rather than
  incidental cleanup.
- `uv` is recommended when available, but a working existing virtual environment
  and pip workflow may be used.
- Ruff is the preferred formatter and linter when configured. Apply it to changed
  code and avoid mass-formatting unrelated historical files.
- mypy is recommended first for pure financial and valuation modules. Adopt it
  incrementally rather than requiring an immediate repository-wide strict pass.

## Testing

- Use pytest.
- Every new or changed financial calculation must have deterministic tests.
- Mock network APIs, external providers, the filesystem, and other unstable
  dependencies in the normal test suite.
- Live-data checks may supplement deterministic tests but must not be required for
  ordinary test success.
- Add UI tests when Streamlit behavior, session state, controls, or displayed
  financial results change materially.
- Preserve existing regression tests unless product behavior intentionally
  changes. Update expectations explicitly and explain the change.
- Do not create tests solely to satisfy a numeric coverage target.
- Keep generated test output and caches out of version control.

## Performance

- Optimize based on profiling or an observed bottleneck.
- This application is usually network-bound. Prefer caching, deduplicated provider
  requests, and lazy loading over premature computational optimization.
- Do not parallelize provider calls blindly; consider rate limits, cache behavior,
  reproducibility, and provider terms.
- Use vectorization for genuinely large tabular operations when it improves both
  clarity and performance. Simple operations on a few annual or quarterly periods
  do not require special optimization.
- Never alter benchmarks or tests to conceal a regression.

## Security

- Never store API keys, passwords, tokens, or credentials in source code.
- Read secrets from environment variables or an ignored local `.env` file.
- Ensure `.env` and generated credential files are ignored by Git.
- Never log secrets or URLs containing API keys.
- Do not commit user data or personally identifiable information.

## Version Control and Scope

- Preserve unrelated user changes in a dirty worktree.
- Do not use destructive Git commands unless explicitly authorized.
- Keep changes narrowly scoped and use clear, descriptive commit messages.
- Do not commit credentials, generated caches, debug output, commented-out code,
  or breakpoints.
- Before committing, inspect the diff and confirm that no unrelated files changed.

## Verification Before Committing

Choose verification proportional to the change. For production code changes,
normally confirm:

- relevant focused tests pass;
- the complete deterministic pytest suite passes when practical;
- changed Python files compile;
- formatting and lint checks pass when the project tools are configured;
- type checks pass for modules currently covered by mypy;
- `git diff --check` passes;
- no hardcoded credentials, debug statements, or unrelated edits were introduced.

Documentation-only changes do not require the full application test suite unless
they affect executable examples or configuration.

## Company Profile Updates

Whenever a Company Profile is added or updated:

1. Keep it as a research candidate. Do not automatically review or apply it to
   the Base DCF.
2. Document evidence, assumption confidence, and model limitations.
3. Run the relevant focused tests, followed by the full deterministic test suite.
4. Start or restart the local Streamlit application with the network access
   required by its live financial-data providers.
5. Open `http://localhost:8501`, select the affected ticker, and verify that the
   Company Profile, fundamentals, evidence, and DCF sections render without an
   unexpected data-loading warning.
6. Leave the local application running and provide the preview link so the user
   can review the candidate.

## Guiding Principle

Prioritize transparent financial reasoning, clarity, and maintainability over
cleverness or rigid adherence to generic tooling rules.
