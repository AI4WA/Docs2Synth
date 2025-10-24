# Repository Guidelines

## Project Structure & Module Organization
- `docs2synth/` hosts the Python package; subpackages such as `agent/`, `datasets/`, `qa/`, `retriever/`, `rag/`, `integration/`, and `preprocess/` map to pipeline stages exposed by `docs2synth.cli`.
- `tests/` mirrors these modules with `test_*.py` suites, while `data/` and `logs/` store sample corpora and run artefacts used during local experimentation.
- MKDocs sources live in `docs/` and are built via `mkdocs.yml`; generated coverage HTML exports land in `htmlcov/`.
- Reuse `config.example.yml` when introducing new environment presets and keep helper scripts inside `scripts/`.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` sets up the editable environment with linting, testing, and documentation dependencies; add `,[mcp]` when developing the provider.
- `docs2synth --help` or subcommands like `docs2synth datasets list` verify the CLI entry point after changes.
- `pytest` runs the default test matrix; add `--cov=docs2synth --cov-report=term-missing` before publishing to match CI.
- `mkdocs serve` previews documentation locally; `mkdocs build` must succeed before doc-focused pull requests.
- `docs2synth-mcp stdio` launches a local MCP provider over stdio (useful for agent integration tests); `docs2synth-mcp http --host 0.0.0.0 --port 8000` starts the FastAPI-backed HTTP endpoint.

## Coding Style & Naming Conventions
- Format Python code with `black` (line length 88) and keep imports sorted with `isort`’s Black profile; run both before raising a PR.
- Use 4-space indentation, `snake_case` for functions and modules, and `CamelCase` for classes; CLI commands should remain short verbs (for example, `datasets list`).
- Annotate new public functions with type hints where practical and prefer explicit exceptions over silent `pass`.

## Testing Guidelines
- Pytest discovers files named `test_*.py`, classes `Test*`, and functions `test_*`; mirror the package path so tests exercise the matching component.
- Maintain or improve coverage; HTML reports are written to `htmlcov/index.html` for deeper reviews.
- Provide representative fixtures for sample documents and retriever payloads, and isolate integrations that touch external services behind marks that CI can skip.

## Commit & Pull Request Guidelines
- Follow the existing history by writing imperative, concise commit titles (for example, `add layout retriever helper`); squash fixup commits locally before pushing.
- Reference related issues in the body (`Closes #42`) and list manual checks (`pytest`, `mkdocs build`, CLI smoke tests).
- Pull requests should outline the problem, the solution, and any configuration updates; attach screenshots for docs or CLI UX tweaks and note follow-up tasks if scope was deferred.

## Documentation & Configuration Tips
- Update `docs/` alongside feature work so the MkDocs site reflects new pipelines or flags.
- When introducing config options, document defaults in `config.example.yml` and mention migration steps in the PR description to help downstream agents stay in sync.
