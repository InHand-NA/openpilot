# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openpilot is an operating system for robotics by comma.ai. It upgrades the driver assistance system (ACC + ALC) in 300+ supported cars. Runs on comma 3X hardware (ARM/aarch64) and x86_64 for development.

## Build & Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Build (SCons-based, compiles C/C++ and Cython)
scons -j$(nproc)

# Run all tests (pytest with xdist parallel execution)
pytest

# Run tests for a specific path, skip slow tests
pytest selfdrive/ -m "not slow"

# Run a single test file
pytest selfdrive/controls/tests/test_alerts.py

# Lint (ruff + codespell + ty type checker)
scripts/lint/lint.sh
scripts/lint/lint.sh --fast    # skip slow checks (ty, codespell)

# Ruff only
ruff check .
ruff format .

# Launch openpilot locally
./launch_openpilot.sh
```

The `op` CLI wrapper (`tools/op.sh`) provides shortcuts: `op build`, `op test`, `op lint`, `op setup`, `op check`.

## Architecture

**Inter-process communication**: All processes communicate via Cap'n Proto messages over a custom message queue (`cereal/` for schemas, `msgq/` for transport). Understanding `cereal` message types is essential for working across subsystems.

**Key subsystems**:
- `selfdrive/` — Core driving logic: vehicle controls (`controls/`), neural network models (`modeld/`), CAN communication (`pandad/`), localization (`locationd/`), driver monitoring (`monitoring/`), the main driving daemon (`selfdrived/`), and UI (`ui/`, raylib-based)
- `system/` — Platform services: process manager (`manager/`), camera daemon (`camerad/`), logging (`loggerd/`), remote connectivity (`athena/`), GPS (`ubloxd/`, `qcomgpsd/`), hardware abstraction (`hardware/`)
- `common/` — Shared utilities: parameter store (`params.py`/`params.cc`), logging (`swaglog`), real-time helpers, filters
- `selfdrive/car/` — Per-brand vehicle interfaces; each brand has its own subdirectory with `carcontroller.py`, `carstate.py`, `interface.py`, and `values.py`

**Submodules** (symlinked into root): `panda/` (vehicle hardware interface + safety model in C), `opendbc/` (CAN signal databases), `tinygrad/` (NN inference), `rednose/` (sensor fusion), `msgq/` (message queue), `teleoprtc/` (remote operation). Avoid modifying these directly.

## Code Style & Conventions

- **Python 3.11+**, 2-space indentation, 160-char line length
- **Absolute imports required**: `from openpilot.selfdrive.foo import bar` (not `from selfdrive.foo`)
- **Banned APIs** (enforced by ruff TID251):
  - `unittest` → use `pytest`
  - `time.time` → use `time.monotonic`
  - `pytest.main` → don't use directly
  - `pyray.draw_text` → use font-explicit functions
  - `pyray.measure_text_ex` → use `openpilot.system.ui.lib.text_measure`
  - `pyray.is_mouse_button_pressed/released` → use `Widget._handle_mouse_press/release`
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants

## Testing

- Framework: `pytest` with `xdist` (parallel by default via `-n auto --dist=loadgroup`)
- Default pytest config ignores submodule directories (opendbc/, panda/, tinygrad_repo/, etc.)
- Markers: `@pytest.mark.slow` (skip with `-m "not slow"`), `@pytest.mark.tici` (device-only tests)
- C/C++ tests: compiled `test_*` binaries run via `selfdrive/test/cpp_harness.py`
- Process replay tests: `selfdrive/test/process_replay/test_processes.py`

## Commit Style

Concise subject prefixed by subsystem in imperative mood: `camerad: fix FPS drop (#123)`

DO NOT commit codes without the user's instruction.

## Safety

The safety model is enforced in `panda/` (C code, MISRA C guidelines). Changes to safety-critical code require extra scrutiny per ISO26262.

## Chat Language
- When you talk to the user, you should always use Chinese.
- When you write documents, please use Chinese also.