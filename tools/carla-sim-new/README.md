# openpilot in CARLA (0.9.15)

本目录提供在 Python 3.12 + CARLA v0.9.15 环境下运行 openpilot 的桥接实现，结构与旧版 0.9.12/0.9.14 基本一致，但依赖当前 inhand-dev 分支的仿真公共组件（tools/sim）。

用法概览：

- 启动 CARLA 服务器（建议 0.9.15，对应 Python 0.9.15 wheel）：
  - Docker 方式（示例，请按需调整镜像 tag 与参数）：
    - `docker run --name carla_sim --rm --gpus all --net=host carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -opengl -nosound -RenderOffScreen -benchmark -fps=20 -quality-level=Low`
- 启动 openpilot 进程（本地）：
  - `./tools/sim/launch_openpilot.sh`
- 启动桥接：
  - `./tools/carla-sim-new/run_bridge.py --host 127.0.0.1 --port 2000 --town Town03 --spawn_point 0 [--high_quality] [--dual_camera]`

说明：

- 桥接管线沿用 `tools/sim` 的通用实现（传感器发布、虚拟 CAN、摄像头 NV12 转换等），仅替换世界层为 CARLA 0.9.15 的 `CarlaWorld`。
- 同步模式（synchronous_mode=True），并保持控制/相机 100Hz/20Hz 的节奏，与旧版一致。
- 分辨率常量 `W,H=1928x1208` 由 `tools/sim/lib/camerad.py` 读取，调整需同步 CARLA 相机 blueprint。

已知注意事项：

- 请确保安装 `carla==0.9.15` 的 Python wheel（与系统/架构匹配），并在虚拟环境中可被 `import carla`。
- 运行在 GPU 上能显著提升实时性；低质量渲染可降低资源消耗。

