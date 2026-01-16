# PC Quickstart: Ubuntu 24 + CARLA

This guide gets openpilot running locally with the CARLA simulator on Ubuntu 24. Keep it simple; exact versions may vary.

## Prerequisites
- NVIDIA GPU recommended for CARLA. Install a recent driver (e.g., 535+).
- Docker installed and running. For GPU access: `sudo apt install -y nvidia-container-toolkit && sudo systemctl restart docker`.

## System packages
```
sudo apt update && sudo apt install -y \
  git build-essential scons docker.io \
  python3.11 python3.11-venv pipx
```
Note: Ubuntu 24 defaults to Python 3.12. openpilot uses Python 3.11; if `python3.11` isn’t available, install via pyenv or your preferred Python manager.

## Poetry and dependencies
```
pipx ensurepath && pipx install poetry
poetry --version
poetry install --sync --with carla
```

## Build native components
```
poetry run scons -j"$(nproc)"
```

## Run the stack
- Terminal 1 (CARLA server in Docker):
```
./tools/sim/start_carla.sh
```
- Terminal 2 (openpilot services for simulation):
```
poetry run ./tools/sim/launch_openpilot.sh
```
- Terminal 3 (bridge between CARLA and openpilot):
```
poetry run ./tools/sim/run_bridge.py --simulator carla
```

Controls: press 2 to engage, 1/2 to adjust speed, S to disengage, r to reset, q to exit.

## Tips
- If you see X11 permission errors with CARLA, run `xhost +local:root` once.
- For lower resource usage, CARLA runs with `-RenderOffScreen -quality-level=Low` in the provided script.
- See more details and alternatives (MetaDrive) in `tools/sim/README.md`.
