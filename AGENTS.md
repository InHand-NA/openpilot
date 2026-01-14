# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `selfdrive/`, platform services in `system/`, shared utilities in `common/`, messaging in `cereal/`, and developer tools in `tools/`.
- Third-party/vendored code sits under `third_party/` and symlinked repos (`opendbc/`, `panda/`, `tinygrad/`, etc.). Avoid modifying these directly.
- Tests are colocated: Python tests follow `test_*.py` across `selfdrive/`, `system/`, and `tools/`; C/C++ tests use `test_*` with a harness in `selfdrive/test/cpp_harness.py`.
- Assets and docs: see `selfdrive/assets/` and `docs/` (mkdocs).

## Build, Test, and Development Commands
- Environment: execute `source .venv/bin/activate` to enter the environment
- Build native/Cython artifacts: `scons -j$(nproc)` (uses `SConstruct`).
- Run tests: `pytest -q` or target a path, e.g., `pytest selfdrive/ -m "not slow"`.
- Lint/format: `ruff check .` and `ruff format .`; spell check with `codespell`.
- Docs: `mkdocs serve` for local docs preview.
- Launch (local): `./launch_openpilot.sh` or use scripts in `scripts/` for specific tasks.

## Coding Style & Naming Conventions
- Python 3.11; line length 160; 2-space indentation (ruff configured). Prefer absolute imports via `openpilot.<module>`.
- Use `pytest` (not `unittest`). Prefer `time.monotonic()` over `time.time()`.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Framework: `pytest` with xdist; default options in `pyproject.toml` (parallel `-n auto`).
- Markers: `slow`, `tici`. Example: skip slow tests locally with `-m "not slow"`.
- Discovery: Python `test_*.py`; C/C++ `test_*` compiled + run via the provided harness.
- Aim to cover new logic; add regression tests alongside changed modules.

## Commit & Pull Request Guidelines
- Commits: concise subject prefixed by subsystem, e.g., `camerad: fix FPS drop (#123)`; imperative mood.
- PRs: clear description, link issues, include before/after or screenshots for UI changes. Ensure CI passes (build, lint, tests) and update docs/tests as needed.
- Use PR templates in `.github/PULL_REQUEST_TEMPLATE/` (selfdrive, system, ui, tools, docs) and pick the closest match.

## Security & Configuration Tips
- Do not commit secrets, logs, or large binaries outside LFS-managed paths. Respect `.gitignore` and `.gitattributes`.
- Changes to symlinked/vendor repos should be proposed upstream or mirrored in their respective subrepos.

## Chat Language
- When you talk to the user, you should always use Chinese.
- When you write documents, please use Chinese also.
