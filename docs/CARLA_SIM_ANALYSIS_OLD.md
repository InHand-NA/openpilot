# CARLA 模拟器实现分析

本文对本项目中 CARLA 模拟器的集成与实现进行技术分析，聚焦工作原理、核心数据流与关键源码位置，帮助读者快速理解“CARLA ↔ openpilot”桥接机制以及传感器/控制链路的端到端映射。

NOTE: 本文基于Openpilot v0.9.5代码实现进行分析，carla版本为0.9.12. 
## 总览与工作原理

- 三进程协同：
  - CARLA 服务器（Docker，GPU 加速），由脚本启动，提供仿真世界、车辆实体与各类传感器接口（相机、IMU、GNSS 等）。
  - openpilot 栈（一组服务进程），通过 cereal/messaging 与 VisionIPC 接收传感器数据、发布控制指令。
  - Bridge 进程（tools/sim/run_bridge.py），在 CARLA 与 openpilot 之间做协议与数据的适配：
    - 从 CARLA 读取传感器数据并封装为 openpilot 期望的消息（加速度计、陀螺仪、GPS、摄像头 YUV 帧等）。
    - 将 openpilot 输出的纵向/横向控制（加速度、制动、转角）转换并施加到 CARLA 车辆。
    - 同时支持键盘/手柄注入人工输入，支持单/双路相机与高/低质量渲染。

- 调度与时序：
  - 控制主循环 100 Hz；相机发布 20 Hz；CARLA 世界推进 tick 与相机 `sensor_tick=1/20` 对齐，避免感知-控制不同步。
  - Bridge 内部通过 Ratekeeper 精确节拍，定期触发 `world.tick()`、传感器消息上报与状态打印。

- 关键启动流程（推荐）：
  1) 启动 CARLA Docker 服务：`tools/sim/start_carla.sh`
  2) 启动 openpilot 仿真服务：`tools/sim/launch_openpilot.sh`
  3) 启动桥接：`tools/sim/run_bridge.py --simulator carla`

参考：`docs/QUICKSTART_PC_CARLA.md:1`，`tools/sim/README.md:1`，`tools/sim/start_carla.sh:1`

## 架构与数据流

- 入口与分发：
  - `tools/sim/run_bridge.py:1` 解析参数后，根据 `--simulator` 选择 `CarlaBridge` 或 `MetaDriveBridge`，并启动输入轮询线程（键盘/手柄）。
  - `CarlaBridge` 继承 `SimulatorBridge`，只需实现 `spawn_world()` 创建 CARLA 世界包装体。

- 抽象基类与主循环：
  - `tools/sim/bridge/common.py:1` 定义 `SimulatorBridge` 和 `World` 抽象：
    - `World` 需提供 `apply_controls/tick/read_sensors/read_cameras/close/reset`。
    - 主循环周期性：
      1) 读取人工输入与 openpilot 控制输出（`controlsState`/`carControl`）。
      2) 组合手动与自动控制，调用 `world.apply_controls(...)` 下发到 CARLA。
      3) 读取传感器，填充 `SimulatorState` 并通过 `SimulatedSensors` 发布到 openpilot。
      4) 每 `TICKS_PER_FRAME` 帧推进一次 `world.tick()` 并抓取相机帧。

- CARLA 侧世界封装：
  - `tools/sim/bridge/carla/carla_bridge.py:1` 在 `spawn_world()` 中创建 CARLA Client 并返回 `CarlaWorld`。
  - `tools/sim/bridge/carla/carla_world.py:1` 完成 CARLA 场景构建、车辆与传感器挂载、控制与读数实现。

- 传感器与视觉链路：
  - `tools/sim/lib/simulated_sensors.py:1` 将 CARLA 读数映射为 cereal 消息：
    - IMU（accelerometer/gyroscope）、GPS（gpsLocationExternal）、驾驶员监控假数据、外设状态。
    - 相机通过 `tools/sim/lib/camerad.py:1` 转换 RGB → NV12（OpenCL 内核 `tools/sim/rgb_to_nv12.cl`）并用 VisionIPC 发布。

- 车辆与 CAN 总线：
  - `tools/sim/lib/simulated_car.py:1` 模拟 Honda Civic 2016 的 CAN 帧，周期发布速度、转角、按钮、IGN 等信号，满足 openpilot 汽车接口期望。

## 核心源码解析

### 1) 运行入口与参数

- `tools/sim/run_bridge.py:1`
  - 关键参数：
    - `--simulator carla` 切换到 CARLA。
    - `--host/--port/--town/--spawn_point` 指定 CARLA 服务器与地图/出生点。
    - `--high_quality/--dual_camera` 控制渲染层与双目相机。
  - 根据 `--joystick` 选择输入轮询源；默认键盘映射详见 `tools/sim/lib/keyboard_ctrl.py:1`。

### 2) Bridge 抽象与主循环

- `tools/sim/bridge/common.py:1`
  - `SimulatorBridge._run()` 初始化：
    - `self.world = self.spawn_world()`
    - `SimulatedCar`（CAN+Panda 状态发布）与 `SimulatedSensors`（IMU/GPS/相机/DM/外设）各自独立线程/节拍。
    - 预热 `world.tick()` 以避免初期帧落后。
  - 控制与参与：
    - 读取 `carControl.actuators`，将线加速度映射为 `throttle_out/brake_out`，将 `steeringAngleDeg` 作为转向角输入给 `world.apply_controls()`。
    - 若 `controlsState.engageable` 且从未成功接管，向 `cruise_button` 注入一次 `DECEL_SET` 以快速进入接管状态。
  - 频率：主循环 100 Hz；`TICKS_PER_FRAME=5` → `world.tick()` 约 20 Hz。

### 3) CARLA 世界实现

- `tools/sim/bridge/carla/carla_world.py:1`
  - 地图与环境：
    - `client.load_world(town, map_layers=...)`，根据 `--high_quality` 选择 `MapLayer.All` 或地面/墙/贴花的低质量组合；`fixed_delta_seconds=0.01`；天气 `ClearSunset`。
  - 车辆与物理：
    - 选取 `vehicle.tesla.*` 蓝图并 `role_name=hero`；可通过 `--spawn_point` 指定出生点。
    - 物理属性：质量 2326kg、扭矩曲线、换挡时延 0；保存前轮 `max_steer_angle` 并设定 `steer_ratio=15` 用于角度缩放。
  - 传感器：
    - 相机：RGB，分辨率取 `W,H`，FOV=40（主路），可选 FOV=120（超广）；`sensor_tick=1/20` 与主循环 tick 对齐；低质量时关闭后处理。
    - IMU：`sensor.other.imu`，`sensor_tick=0.01`；GNSS：`sensor.other.gnss`；并设置 `Params().put_bool("UbloxAvailable", True)` 以匹配 openpilot GPS 期望。
  - 控制下发：
    - `apply_controls(steer_angle, throttle_out, brake_out)` 将角度（deg）映射到 CARLA `[-1,1]`：`steer_carla = -steer_angle / (max_steer_angle * steer_ratio)` 并裁剪；写入 `VehicleControl`。
  - 读数采集：
    - IMU 三轴、角速度、航向角；车辆 `get_velocity()`；根据 `spawn_point` 支持 `reset()`。
    - 相机采用回调 `camera.listen()` 异步写入 `road_image/wide_road_image`，`read_cameras()` 为空实现。

### 4) 传感器与视觉发布

- `tools/sim/lib/simulated_sensors.py:1`
  - IMU：5 次/循环发送加速度计与陀螺仪；时间戳使用 `logMonoTime`（后续可对齐传感器时间戳）。
  - GPS：从 `SimulatorState.gps` 取经纬高度，从 CARLA 速度向量做 NED 坐标变换（CARLA 的北向为 `-Y`），并设置 `SensorSource.ublox`。
  - 相机：持有 `Camerad`，从 `world` 的 `road_image`（与可选 `wide_road_image`）取帧，调用 OpenCL 核心 `rgb_to_nv12` 转换，再通过 VisionIPC 发布 YUV 帧。
  - 驾驶员监控与外设：周期注入固定的“面部检测成功/不分心”等假数据与 panda 外设状态，满足下游依赖。

- `tools/sim/lib/camerad.py:1`
  - 创建 VisionIPC 缓冲区（路、超广各 5 帧）；OpenCL 上下文/队列；编译 `tools/sim/rgb_to_nv12.cl` 内核实现高效 RGB→NV12 转换；按帧序号发布并同步 `roadCameraState/wideRoadCameraState` 元数据。

### 5) 车辆与 CAN 融合

- `tools/sim/lib/simulated_car.py:1`
  - 使用 opendbc 的 `CANPacker` 构造 Honda Civic 2016 所需 CAN 帧，覆盖车速、方向盘角度、踏板、巡航按钮、点火、HUD 等；周期插入 radar 总线心跳。
  - `pandaStates` 周期（2Hz）发布，设置 `controlsAllowed=True` 与 `safetyModel=hondaNidec` 等，匹配 openpilot 车辆安全模型与策略。
  - 该层保证即便物理世界来自 CARLA，openpilot 的车机接口仍保持“像在真车上一样”的消息契约。

## 时序与频率对齐

- 主循环 100 Hz；`CarlaBridge.TICKS_PER_FRAME=5` → `world.tick()` 约 20 Hz，对齐相机 `sensor_tick=1/20`，降低视觉-控制抖动。
- IMU 高频（每循环 5 次）、GPS 高频（每循环 10 次，带 NED 速度），确保感知与定位链路稳定。

## 启动脚本与依赖

- `tools/sim/start_carla.sh:1`
  - 拉取 `carlasim/carla:0.9.14`，GPU 直通、host 网络、X11 映射；以 `-RenderOffScreen -quality-level=Low -fps=20` 运行，资源占用更可控。
  - 需要 `nvidia-container-toolkit`，脚本提供一键安装提示。

- 依赖安装（CARLA Python API）：
  - `pyproject.toml:167` 与 `poetry.lock` 引入 `carla==0.9.14` wheel（x86_64/Linux）；通过 `poetry install --with carla` 启用。
  - 参考快速指南：`docs/QUICKSTART_PC_CARLA.md:1`。

## 测试与验证

- `tools/sim/tests/test_carla_bridge.py:1`
  - 启动/停止 CARLA Docker，创建 `CarlaBridge` 并配合 `test_sim_bridge.py`：
    - 启动 openpilot 管理器与 bridge，确保所有必需进程运行、无阻断类 CarEvent。
    - 注入“巡航设定”按键以自动接管，连续检测 `controlsState.active` 达到阈值。

## 关键可配置项

- 运行时参数（`tools/sim/run_bridge.py:1`）：`--host/--port/--town/--spawn_point/--dual_camera/--high_quality`。
- 分辨率常量：`tools/sim/lib/common.py:1` 中 `W,H=1928x1208`（由 camerad 引用），如需变更需同步CARLA相机与 VisionIPC 缓冲区设置。
- 控制缩放：`carla_world.py` 中 `steer_ratio=15` 与 `max_steer_angle`；纵向控制由 openpilot 加速度输出线性分拆至油门/制动（`common.py` 主循环处）。

## 已知限制与优化建议

- 性能：CARLA 对 GPU/CPU 要求高，默认低质量渲染；如需画质，可启用 `--high_quality` 并在 `start_carla.sh` 调整 CLI 参数，但需权衡实时性。
- 时间戳：IMU/GPS 使用 `logMonoTime` 或现读时刻；若进行严格融合评估，可将 CARLA 传感器原始时间戳贯穿到 cereal 消息。
- 车辆动力学：当前选用 Tesla 蓝图 +简单物理参数，若验证特定车型控制器，建议按车型校准质量/转向比/轮胎模型并对齐 CAN 信号学。
- 坐标系：速度做了 CARLA→NED 的轴向转换（北向为 -Y），多源融合或高精地图时需统一坐标与地理投影（`tools/sim/lib/common.py:1` 的 `GPSState.from_xy` 仅做平面近似）。

## 代码索引与参考

- 入口与文档：
  - `tools/sim/run_bridge.py:1`
  - `tools/sim/README.md:1`
  - `docs/QUICKSTART_PC_CARLA.md:1`

- 核心桥接与世界：
  - `tools/sim/bridge/common.py:1`
  - `tools/sim/bridge/carla/carla_bridge.py:1`
  - `tools/sim/bridge/carla/carla_world.py:1`

- 传感器与视觉：
  - `tools/sim/lib/simulated_sensors.py:1`
  - `tools/sim/lib/camerad.py:1`
  - `tools/sim/rgb_to_nv12.cl:1`

- 车辆/总线：
  - `tools/sim/lib/simulated_car.py:1`

- 启动与测试：
  - `tools/sim/start_carla.sh:1`
  - `tools/sim/tests/test_carla_bridge.py:1`
  - `tools/sim/tests/test_sim_bridge.py:1`

以上分析覆盖 CARLA 集成的整体架构、关键链路与源码要点，可作为二次开发（例如加传感器、换车型、调时序/性能）时的快速参考。


## 如何运行

- 启动 CARLA 0.9.15（示例 Docker）
`start_carla.sh (line 1)`

- 启动 openpilot（保持摄像头/日志等仿真块）

`launch_openpilot.sh (line 1)`
- 启动桥接
`run_bridge.py` (line 1)
示例：`run_bridge.py --host 127.0.0.1 --port 2000 --town Town03 --spawn_point 0 --dual_camera`
键盘控制见 keyboard_ctrl.py (line 1)（1/2/3巡航，wasd手动，r重置，i点火，q退出）
