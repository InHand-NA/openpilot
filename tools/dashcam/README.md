# dashcam — 基于 Carla 仿真的 openpilot 感知可视化工具

dashcam 是一个将 openpilot 感知管线（modeld + calibrationd）接入 Carla 模拟器的端到端工具。它在仿真环境中运行真实的 openpilot 神经网络模型，渲染车道线、路边沿、前车检测等感知结果，并支持在线标定和视频录制。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                   主进程 (run.py)                         │
│                                                           │
│  Carla 客户端        VisionIPC 服务端       可视化渲染     │
│  (carla_world.py)    (双目 YUV420)         (visualizer.py)│
│  - 双目 RGB 相机     - road 帧流            - 车道线       │
│                     - wide 帧流             - 路边沿       │
│  - 自车 + NPC                               - 前车标记     │
│                                              - 标定面板     │
│  cereal 消息总线                                          │
│  发布: carState, deviceState, [liveCalibration]           │
│  订阅: modelV2, liveCalibration                           │
└────────────┬────────────────────┬─────────────────────────┘
             │                    │
             v                    v
     ┌──────────────┐    ┌────────────────┐
     │   modeld     │    │  calibrationd  │
     │   (CUDA)     │    │  (可选子进程)   │
     │              │    │                │
     │ 输入:        │    │ 输入:          │
     │  VisionIPC   │    │  cameraOdometry│
     │  liveCalib   │    │  carState      │
     │              │    │                │
     │ 输出:        │    │ 输出:          │
     │  modelV2     │    │  liveCalib     │
     │  cameraOdom  │    │                │
     └──────────────┘    └────────────────┘
```

**消息流**：Carla 帧 → VisionIPC → modeld 推理 → modelV2 感知输出 → visualizer 渲染叠加 → 显示/录制

## 文件说明

| 文件 | 说明 |
|------|------|
| `run.py` | 主入口：编排子进程、Carla 连接、消息发布/订阅、可视化循环 |
| `camerad.py` | VisionIPC 服务端：支持双目和单广角（wide-road-only）两种模式 |
| `carla_world.py` | Carla 环境管理：自车/NPC 生成、相机挂载、帧采集 |
| `visualizer.py` | 感知渲染：车道线多边形、路边沿、前车三角标记、标定进度面板 |
| `calibrationd.py` | 在线标定：从视觉里程计估计相机姿态（pitch/yaw）和高度 |
| `start_carla.sh` | 启动 Carla 0.9.16 Docker 容器（NVIDIA GPU、Epic 画质） |

## 前置条件

| 依赖 | 要求 |
|------|------|
| Carla | 0.9.16（Docker 镜像 `carlasim/carla:0.9.16`） |
| GPU | NVIDIA（CUDA 用于 modeld 推理，推荐 RTX 4090） |
| Docker | 需要 `--runtime=nvidia` 支持 |
| Python | 3.11+，openpilot 虚拟环境已激活 |

## 快速开始

```bash
# 1. 激活 openpilot 环境
source .venv/bin/activate

# 2. 启动 Carla 服务端（首次会自动拉取 Docker 镜像）
./tools/dashcam/start_carla.sh

# 3. 运行 dashcam（另开终端）
python tools/dashcam/run.py --perfect-cam --high-quality
```

按 `q` 或 `ESC` 退出。

## 命令行参数

### 连接与场景

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | Carla 服务器地址 |
| `--port` | `2000` | Carla 服务器端口 |
| `--town` | `Town04_Opt` | Carla 地图名称 |
| `--spawn-point` | `16` | 自车出生点索引 |
| `--num-npc` | `20` | NPC 车辆数量 |

### 相机姿态

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera-pitch` | `5.0` | 相机俯仰角（度），正值=向下 |
| `--camera-yaw` | `3.0` | 相机偏航角（度），正值=向左 |
| `--camera-height` | `1.13` | 相机离路面高度（米） |
| `--perfect-cam` | - | 理想安装：pitch=0, yaw=0 |

### 标定模式

| 参数 | 说明 |
|------|------|
| （默认） | **已知姿态模式**：使用命令行指定的 pitch/yaw 作为固定标定值 |
| `--online-calib` | **在线标定模式**：启动 calibrationd 子进程，从视觉里程计实时估计姿态 |

### 运行控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--high-quality` | - | 高画质（完整地图层+后处理） |
| `--no-display` | - | 无头模式，不显示窗口 |
| `--fast` | - | 全速运行，不做帧率限制 |
| `--save-video` | `''` | 保存可视化为 MP4 文件 |
| `--max-frames` | `0` | 最大帧数（0=无限） |
| `--wide-road-only` | - | 单广角相机模式（仅 WIDE_ROAD 流，modeld 用 ecam intrinsics） |
| `--road-only` | - | 单窄角相机模式（仅 ROAD 流，modeld 用 fcam intrinsics） |

## 使用示例

```bash
# 理想相机 + 高画质（最常用）
python tools/dashcam/run.py --perfect-cam --high-quality

# 在线标定模式：模拟相机歪了 5° 俯仰 + 3° 偏航
python tools/dashcam/run.py --online-calib --camera-pitch 5 --camera-yaw 3

# 录制视频（100 帧后自动停止）
python tools/dashcam/run.py --perfect-cam --save-video output.mp4 --max-frames 100

# 全速无显示模式（用于性能测试）
python tools/dashcam/run.py --perfect-cam --no-display --fast

# 自定义场景：Town03 地图，50 辆 NPC
python tools/dashcam/run.py --perfect-cam --town Town03 --num-npc 50

# 单广角相机模式（与 openpilot 仅 WIDE_ROAD 模式一致）
python tools/dashcam/run.py --perfect-cam --wide-road-only

# 单窄角相机模式（仅 ROAD 流）
python tools/dashcam/run.py --perfect-cam --road-only
```

## 在线标定系统

使用 `--online-calib` 时，calibrationd 子进程从 modeld 的视觉里程计（`cameraOdometry`）实时估计相机姿态。

### 标定条件

标定仅在以下条件**同时满足**时更新：

| 条件 | 阈值 | 说明 |
|------|------|------|
| 车速 | > 24 km/h | `carState.vEgo > MIN_SPEED_FILTER` |
| 视觉速度 | > 1.3 m/s | `cameraOdometry.trans[0]`（仿真模式放宽） |
| 偏航角速度 | < 2 deg/s | 要求直线行驶 |

### 标定进度

- 每 100 个有效样本组成一个 **block**
- 累计 **5 个 block** 后状态变为 `calibrated`
- 滑动窗口最多维持 50 个 block
- UI 面板显示实时进度条、百分比和状态

### 标定状态

| 状态 | 颜色 | 含义 |
|------|------|------|
| `UNCALIBRATED` | 黄色 | 有效 block < 5，标定进行中 |
| `CALIBRATED` | 绿色 | 有效 block >= 5，pitch/yaw 在合法范围内 |
| `INVALID` | 红色 | 有效 block >= 5，但 pitch/yaw 超出范围 |
| `RECALIBRATING` | 橙色 | 检测到安装变化，正在重新标定 |

## 可视化渲染

渲染管线对齐 openpilot UI（`selfdrive/ui/onroad/model_renderer.py`）：

- **车道线**：绿色填充多边形，宽度 = `0.025 * prob` 米，alpha = `clip(prob, 0, 0.7)`
- **路边沿**：红色填充多边形，固定宽度 0.025 米，alpha = `clip(1 - std, 0, 1)`
- **前车标记**：双层三角形（外层黄色 glow + 内层红色 chevron），尺寸和透明度随距离/相对速度变化
- **距离裁剪**：仅渲染 10m ~ 100m 范围内的点，末端做插值平滑

最终输出分辨率为 2160×1080（与 openpilot UI 一致），原始 1928×1208 帧经 1.12x 缩放 + 居中裁剪。

## 启动 Carla 服务端

```bash
# 前台运行（看到 Carla 输出日志）
./tools/dashcam/start_carla.sh

# 后台运行
DETACH=1 ./tools/dashcam/start_carla.sh

# 停止
docker kill $(docker ps -q --filter ancestor=carlasim/carla:0.9.16)
```

Carla 服务端监听 `localhost:2000`，使用 Epic 画质 + 离屏渲染模式。

## NPC 车辆管理

- NPC 使用 Carla Traffic Manager 的 autopilot 自动驾驶
- 休眠机制：距离自车 > 150m 的 NPC 进入休眠
- 自动重生：休眠的 NPC 会被传送回自车 25m ~ 100m 范围内
- 保证自车周围始终有交通流量
