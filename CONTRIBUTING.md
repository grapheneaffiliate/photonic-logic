# Contributing

Thanks for your interest in improving Photonic Logic!

## Quick start

1. Fork the repo and create a feature branch:
   - git checkout -b feat/your-feature

2. Set up the dev environment:
   - python -m pip install -e .[dev]
   - pre-commit install  # optional but recommended

3. Run quality checks locally:
   - ruff check .
   - black --check .
   - mypy src
   - pytest -q

4. Commit style
   - Use small, focused commits with clear messages, e.g.:
     - ci(actions): add Python CI workflow
     - tests(core): add passivity and trend checks
     - feat(cli): add --out report option
   - Include before/after notes in the PR description if behavior changes.

5. Open a Pull Request
   - Target branch: main
   - Fill in a concise description and motivation
   - Link related issues if any

## Scope guidance

- Keep hardware IO abstracted; provide simulators/mocks for CI.
- Guard physics routines with passivity and stability tests.
- Prefer non-breaking CLI changes; add flags over renaming commands.
- Keep reproducibility in mind (seed randomness; write deterministic outputs).

## Code style

- Python 3.10+
- Type hints required in public APIs
- Formatting: black (line length 100)
- Lint: ruff (imports ordered)
- Types: mypy (config in pyproject.toml)

## Testing

- Use pytest
- Keep tests fast and deterministic
- Avoid hardware dependencies; use pure software mocks
- Add regression tests for discovered bugs

## Licensing

By contributing, you agree your contributions are licensed under the repository’s MIT License.
