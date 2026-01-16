# Repository Guidelines

## Project Structure & Modules
- Core code lives in `selfdrive/`, `system/`, and `tools/`. Supporting libs in `common/`, protocol data in `cereal/`, CAN databases in `opendbc/`, hardware in `panda/`.
- Tests are colocated: see `selfdrive/test/`, `selfdrive/**/tests/`, `system/**/tests/`, `tools/lib/tests/`, and `tools/replay/`.
- Docs and CI live in `docs/` and `.github/`. Build logic is in `SConstruct` and `site_scons/`.

## Build, Test, and Development
- Setup (Python 3.11): `poetry install` (add `--with carla` for simulator deps).
- Lint/Type-check: `poetry run ruff check .` and `poetry run mypy --explicit-package-bases .`.
- Build native components: `poetry run scons -j$(nproc)`.
- Run tests (parallel by default): `poetry run pytest -m "not tici"`.
- Simulation (CARLA): see `tools/sim/README.md` and `docs/QUICKSTART_PC_CARLA.md`; typical flow after install is `poetry run python tools/sim/run_bridge.py --simulator carla`.

## Coding Style & Naming
- Python: 2-space indentation, max line length 160 (`tool.ruff`). Prefer absolute imports under the `openpilot.*` namespace (ruff bans bare `selfdrive`, `common`, etc.).
- C/C++: enforced via `cppcheck`, `cpplint` (line length 240) and `.clang-tidy`.
- Keep modules small and directory-scoped; mirror runtime services under `selfdrive/<service>` and tests as `test_*.py`.
- Run `pre-commit install` and use `pre-commit run -a` before committing.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-xdist`, `pytest-cov`. Tests discovered in the paths configured in `pyproject.toml` (`[tool.pytest.ini_options].testpaths`).
- Naming: files `test_*.py`, C++ tests `test_*` per `cpp_files`.
- Markers: use `@pytest.mark.slow` for long tests and `@pytest.mark.tici` for device-only tests. Example: `pytest -m "not slow and not tici"`.

## Commit & Pull Requests
- Commits: concise, imperative subject; optionally prefix scope (e.g., `ui: add purple prime def`) and reference PRs (`(#12345)`).
- PRs: include a clear description and verification steps. For car-related changes, follow templates in `.github/PULL_REQUEST_TEMPLATE/` and include routes, logs, and screenshots as requested.

## Security & Configuration
- Do not commit secrets or personal data. See `SECURITY.md`.
- Scripts `launch_openpilot.sh`/`launch_env.sh` target device environments; for PC use tests and CARLA simulation.

## Agent-Specific Notes
- These guidelines apply repo-wide. When modifying code, keep changes minimal, respect existing structure, and update tests alongside code.
