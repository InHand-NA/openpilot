# dashcam — 基于 Carla 仿真的 openpilot 感知数据采集与模型训练工具

dashcam 是一个将 openpilot 感知管线（modeld + calibrationd）接入 Carla 模拟器的端到端工具。它在仿真环境中运行真实的 openpilot 神经网络模型，渲染车道线、路边沿、前车检测等感知结果，并支持在线标定、视频录制、**训练数据采集**和**模型训练**。

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

**自训练模型模式**（`--custom-model`）：绕过 modeld/VisionIPC/cereal，在主进程内直接用 onnxruntime 对 RGB 帧执行 ONNX 推理，结果转为 modelV2 消息后传递给 visualizer 渲染。

```
Carla RGB 帧 (1208×1928)
  → warp_image (512×256)
  → YUV420 6ch (6×128×256)
  → 帧对拼接 (12×128×256)
  → onnxruntime 推理
  → MDN/BCE 解码
  → cereal modelV2 消息
  → visualizer 渲染
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `run.py` | 主入口：编排子进程、Carla 连接、消息发布/订阅、可视化循环 |
| `camerad.py` | VisionIPC 服务端：支持双目和单广角（wide-road-only）两种模式 |
| `carla_world.py` | Carla 环境管理：自车/NPC 生成、相机挂载、帧采集 |
| `visualizer.py` | 感知渲染：车道线多边形、路边沿、前车三角标记、标定进度面板 |
| `calibrationd.py` | 在线标定：从视觉里程计估计相机姿态（pitch/yaw）和高度 |
| `data_recorder.py` | 训练数据记录：逐帧保存 RGB + GT 标签为压缩 NPZ 文件 |
| `lane_ground_truth.py` | 车道线真值提取：从 Carla 地图 API 获取 4 条车道线 + 2 条路边沿 |
| `lead_ground_truth.py` | 前车真值提取：检测前方车辆并生成时序预测轨迹 |
| `pose_ground_truth.py` | 自车运动真值：计算帧间平移速度和角速度 |
| `view_npz.py` | NPZ 可视化工具：在 warp 图像上叠加所有 GT 标注 |
| `infer.py` | 自训练 ONNX 模型进程内推理：warp → YUV420 → onnxruntime → modelV2 消息 |
| `warp_example.py` | Warp 示例：演示训练时原始图像到模型输入空间的透视变换 |
| `collect_multi_height.sh` | 批量数据采集：遍历多相机高度和多地图 |
| `start_carla.sh` | 启动 Carla 0.9.16 Docker 容器（NVIDIA GPU、Epic 画质） |
| `train/` | 模型训练框架（详见下方「模型训练」章节） |

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
| `--custom-model` | `''` | 自训练 ONNX 模型路径（绕过 modeld，进程内 onnxruntime 推理） |

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

# 使用自训练 ONNX 模型（绕过 modeld，进程内推理）
python tools/dashcam/run.py --perfect-cam --road-only --high-quality \
  --custom-model checkpoints/driving_vision.onnx

# 自训练模型 + 录制视频
python tools/dashcam/run.py --perfect-cam --high-quality \
  --custom-model checkpoints/driving_vision.onnx \
  --save-video custom_output.mp4 --max-frames 500
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

---

## 训练数据采集

### 数据采集流程

```
Carla 模拟器 (0.9.16, 20 FPS)
    ↓
DashcamCarlaWorld (carla_world.py)
    ├─ RGB 相机捕获 (1928×1208)
    ├─ 自车运动跟踪
    └─ NPC 车辆管理
    ↓
真值提取（仅 record-only 模式）
    ├─ LaneGroundTruth  → 4 条车道线 + 2 条路边沿
    ├─ LeadGroundTruth  → 3 个前车候选 × 时序轨迹
    └─ PoseGroundTruth  → 自车 6DoF 运动
    ↓
DataRecorder (data_recorder.py)
    └─ 保存为压缩 NPZ 文件（每帧一个文件）
```

### 采集命令

```bash
# 1. 启动 Carla
./tools/dashcam/start_carla.sh

# 2. 基本采集（理想相机，record-only 最快）
python tools/dashcam/run.py \
  --perfect-cam \
  --record data/training/carla_001 \
  --record-only --fast \
  --max-frames 20000

# 3. 自定义车速范围
python tools/dashcam/run.py \
  --perfect-cam \
  --record data/training/carla_002 \
  --record-only --fast \
  --speed-range 30 120 \
  --max-frames 20000

# 4. 指定相机高度（模拟卡车等不同车型）
python tools/dashcam/run.py \
  --perfect-cam \
  --camera-height 2.4 \
  --record data/training/truck_h2.4 \
  --record-only --fast

# 5. 跳帧采集（每 4 帧保存 1 帧，减少磁盘占用）
python tools/dashcam/run.py \
  --perfect-cam \
  --record data/training/carla_003 \
  --record-skip 4 \
  --record-only --fast

# 6. 切换地图
python tools/dashcam/run.py \
  --perfect-cam \
  --town Town03_Opt \
  --record data/training/town03 \
  --record-only --fast
```

### 采集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--record <dir>` | `''` | 启用数据采集，保存到指定目录 |
| `--record-only` | - | 仅采集模式：禁用 modeld 和可视化，采集速度最快 |
| `--record-skip <N>` | `1` | 每 N 帧保存 1 帧（1 = 全部保存） |
| `--speed-range <min> <max>` | `20 140` | 自车目标速度范围（km/h） |
| `--fast` | - | 全速运行，不做帧率限制 |
| `--max-frames <N>` | `0` | 最大帧数（0 = 无限） |
| `--camera-height <m>` | `1.13` | 相机离路面高度（米） |
| `--perfect-cam` | - | 理想安装（pitch=0, yaw=0） |

### 批量多样化采集

使用 `collect_multi_height.sh` 自动遍历多相机高度和多地图：

```bash
bash tools/dashcam/collect_multi_height.sh
```

默认配置：
- 高度：1.2m, 1.8m, 2.0m, 2.4m, 2.8m
- 地图：Town04_Opt, Town03_Opt, Town06_Opt
- 每组 2000 帧，共 15 组 = 30,000 帧

输出目录结构：
```
data/training/
├── Town04_Opt_h1.2/    # 2000 帧
├── Town04_Opt_h1.8/    # 2000 帧
├── ...
└── Town06_Opt_h2.8/    # 2000 帧
```

### NPZ 数据格式

每个 `.npz` 文件包含单帧的完整图像和标注：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `frame_rgb` | (1208, 1928, 3) | uint8 | 原始 RGB 图像 |
| `lane_lines` | (4, 33, 3) | float32 | 4 条车道线 × 33 采样点 × (x, y, z) |
| `lane_lines_prob` | (4,) | float32 | 4 条车道线的存在概率 [0, 1] |
| `road_edges` | (2, 33, 3) | float32 | 2 条路边沿 × 33 采样点 × (x, y, z) |
| `road_edges_prob` | (2,) | float32 | 2 条路边沿的存在概率 |
| `lead` | (3, 6, 4) | float32 | 3 个前车候选 × 6 时间步 × (x距离, y偏移, 速度, 加速度) |
| `lead_prob` | (3,) | float32 | 3 个前车候选的存在概率 |
| `pose` | (6,) | float32 | 自车运动 [vx, vy, vz, wx, wy, wz] (m/s, rad/s) |
| `road_transform` | (6,) | float32 | 道路变换参数 |
| `rpyCalib` | (3,) | float32 | 相机标定 [roll, -pitch, -yaw] (弧度) |
| `v_ego` | scalar | float32 | 自车速度 (m/s) |
| `camera_height` | scalar | float32 | 相机高度 (m) |
| `camera_pitch` | scalar | float32 | 相机俯仰角 (弧度) |
| `camera_yaw` | scalar | float32 | 相机偏航角 (弧度) |
| `world_pose` | (6,) | float32 | Carla 世界坐标 [x, y, z, roll°, pitch°, yaw°] |
| `town` | str | object | Carla 地图名称 |

**坐标系**：校准坐标系（x=前向, y=右向, z=向下，重力对齐）。车道线和路边沿在 `ModelConstants.X_IDXS`（0m~192m）的 33 个前向距离处采样。

**车道线排列**：
```
[0] far-left    左相邻车道的左边界
[1] near-left   本车道的左边界
[2] near-right  本车道的右边界
[3] far-right   右相邻车道的右边界
```

**前车数据含义**：
- 维度 0（3 个候选）：当前时刻 / 2 秒后 / 4 秒后的最近前车
- 维度 1（6 个时间步）：[0, 2, 4, 6, 8, 10] 秒
- 维度 2（4 个参数）：前向距离 (m), 横向偏移 (m), 绝对速度 (m/s), 加速度 (m/s²)

### 数据验证与可视化

```bash
# 查看单帧 NPZ 数据（在 warp 后的模型输入图上叠加 GT 标注）
python tools/dashcam/view_npz.py data/training/carla_001/000100.npz

# 浏览目录（← → 键翻页）
python tools/dashcam/view_npz.py data/training/carla_001/

# 批量导出为 PNG
python tools/dashcam/view_npz.py data/training/carla_001/ --save-dir outputs/

# 查看 warp 变换效果
python tools/dashcam/warp_example.py data/training/carla_001/000100.npz
```

---

## 模型训练

### 模型概述

基于 openpilot `driving_vision.onnx` 架构的精简版单摄像头视觉模型，保留 LDW（车道偏离预警）和 FCW（前碰撞预警）所需的输出。

```
输入: (B, 12, 128, 256)  — 2 帧 × 6 通道 YUV420
       ↓
骨干: FastViT RepMixer [2,2,6,2], channels [64,128,256,512]
      → DWConv(512→1024) + SE + GAP + FC → 2048 维特征
       ↓
   ┌───────────────────┐    ┌───────────────────┐
   │ Bottleneck Head   │    │ No-Bottleneck Head│
   │ (L2Norm, 512维)   │    │ (512维)           │
   │                   │    │                   │
   │ ├ lane_lines  528 │    │ ├ pose         12 │
   │ ├ lane_lines_prob │    │ └ road_transform  │
   │ │              8  │    │               12  │
   │ ├ road_edges  264 │    └───────────────────┘
   │ ├ lead        144 │
   │ └ lead_prob     3 │
   └───────────────────┘
        总输出: 971 维
```

### 训练框架文件

| 文件 | 说明 |
|------|------|
| `train/config.py` | 模型与训练超参数配置（`ModelConfig` + `TrainConfig`） |
| `train/model.py` | PyTorch 模型定义（FastViT 骨干 + 双路输出 Head） |
| `train/dataset.py` | 数据集加载（帧配对、warp、YUV420 转换、GT 提取、NaN 处理） |
| `train/losses.py` | 损失函数（GaussianNLL + BCEWithLogits + 组合加权 + 逐点 mask） |
| `train/train.py` | 训练脚本（AdamW + CosineAnnealing + warmup + 早停 + CSV 日志） |
| `train/monitor.py` | 训练监控工具（实时指标查看、训练曲线绘制） |
| `train/export_onnx.py` | ONNX 导出（RepConv 重参数化 + onnxruntime 验证 + fp16 转换） |

### 环境准备

训练需要 PyTorch GPU 版本。在 openpilot 虚拟环境中安装：

```bash
source .venv/bin/activate

# 确保 venv 内有 pip
python -m ensurepip

# 安装 PyTorch（根据 CUDA 版本选择，以 CUDA 12.4 为例）
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 开始训练

```bash
# 基本用法：使用已有数据训练
python tools/dashcam/train/train.py \
  --data-dirs data/training/carla_001 \
  --output-dir checkpoints \
  --epochs 100 --batch-size 16 --lr 1e-3

# 启用早停：val_loss 连续 15 个 epoch 无改善则自动终止
python tools/dashcam/train/train.py \
  --data-dirs data/training/carla_001 \
  --epochs 100 --early-stop 15

# 使用多个数据集
python tools/dashcam/train/train.py \
  --data-dirs data/training/carla_001 data/training/carla_002 data/training/town03 \
  --output-dir checkpoints \
  --epochs 100 --early-stop 15

# 大显存 GPU（如 RTX 4090 24GB）可增大 batch_size
python tools/dashcam/train/train.py \
  --data-dirs data/training/carla_001 \
  --batch-size 32 --lr 2e-3

# 从 checkpoint 恢复训练
python tools/dashcam/train/train.py \
  --data-dirs data/training/carla_001 \
  --resume checkpoints/best.pt \
  --epochs 200

# 仅训练不导出 ONNX
python tools/dashcam/train/train.py \
  --data-dirs data/training/carla_001 \
  --no-export
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dirs` | 必填 | 训练数据目录（支持多个） |
| `--output-dir` | `checkpoints` | checkpoint 输出目录 |
| `--epochs` | `100` | 训练总轮数 |
| `--batch-size` | `16` | 批大小 |
| `--lr` | `1e-3` | 初始学习率 |
| `--resume` | - | 从指定 checkpoint 恢复训练 |
| `--no-export` | - | 训练结束不自动导出 ONNX |
| `--early-stop` | `0` | 早停耐心值（0=禁用），val_loss 连续 N 个 epoch 无改善则终止 |

### 训练流程

1. 加载所有数据目录中的 NPZ 文件，按 95/5 比例随机划分训练集/验证集
2. 每帧与前 4 帧配对（temporal_skip = MODEL_RUN_FREQ / MODEL_CONTEXT_FREQ = 4），不跨目录边界
3. 图像预处理：原始 RGB → warp 到 512×256 → YUV420 6 通道 → 归一化到 [-1, 1]
4. GT 标签预处理：lane_lines/road_edges 中的 NaN（山顶遮挡截断点）替换为 0，并生成逐点 valid mask
5. 优化器：AdamW（weight_decay=1e-4），学习率调度：5 epoch 线性 warmup + 余弦退火
6. 梯度裁剪 max_norm=1.0
7. 每 epoch 记录指标到 CSV 日志（`training_log.csv`）
8. 每 5 epoch 保存 checkpoint，持续跟踪最佳验证 loss
9. 可选早停：val_loss 连续 N epoch 无改善则终止
10. 训练完成自动导出 ONNX（fp32 + fp16 两个版本）

### 损失函数

| 输出 | 损失类型 | 权重 | 说明 |
|------|---------|------|------|
| `lane_lines` | GaussianNLL | 1.0 | MDN (μ, log σ)，双重 mask：prob > 0.5（车道级）× valid（逐点，排除 NaN 截断点） |
| `lane_lines_prob` | BCEWithLogits | 1.0 | 概率扩展为 logit 对 [1-p, p] |
| `road_edges` | GaussianNLL | 1.0 | 双重 mask：prob > 0.5 × valid（同上） |
| `lead` | GaussianNLL | 0.5 | 按 lead_prob > 0.5 做 mask |
| `lead_prob` | BCEWithLogits | 1.0 | |
| `pose` | GaussianNLL | 0.2 | 无 mask，每帧都有 |
| `road_transform` | GaussianNLL | 0.2 | 无 mask |

> **NaN 处理**：GT 中 lane_lines 和 road_edges 在远处山顶遮挡处会有 NaN 值。dataset 层将 NaN 替换为 0 并生成逐点 valid mask `(N, 33)`，loss 层在计算 GaussianNLL 时同时使用 prob mask（车道/路边沿级别）和 valid mask（采样点级别），确保 NaN 点不参与梯度计算。

### 训练输出

```
checkpoints/
├── checkpoint_epoch5.pt        # 每 5 epoch 保存
├── checkpoint_epoch10.pt
├── ...
├── best.pt                     # 最佳验证 loss
├── final.pt                    # 最终 epoch
├── driving_vision.onnx         # fp32 ONNX 模型（~76MB）
├── driving_vision_fp16.onnx    # fp16 ONNX 模型（~38MB）
├── training_log.csv            # 逐 epoch 指标日志
└── training_log_curves.png     # 训练曲线图（monitor.py --plot 生成）
```

每个 checkpoint 包含 `model_state_dict`、`optimizer_state_dict`、`epoch`、`val_loss`。

### 训练监控

训练过程中每个 epoch 的指标自动写入 `training_log.csv`，可用 `monitor.py` 实时查看：

```bash
# 一次性查看当前状态
python tools/dashcam/train/monitor.py checkpoints/training_log.csv

# 实时刷新（默认 10 秒间隔）
python tools/dashcam/train/monitor.py checkpoints/training_log.csv --live

# 自定义刷新间隔（30 秒）
python tools/dashcam/train/monitor.py checkpoints/training_log.csv --live --interval 30

# 导出训练曲线图（保存为 PNG）
python tools/dashcam/train/monitor.py checkpoints/training_log.csv --plot
```

监控输出示例：

```
======================================================================
  Epoch: 100    LR: 0.000000    Time/epoch: 227s
  Train loss: -3.8775    Val loss: -4.0582
  Best val:   -4.0582 (epoch 100, 0 epochs ago)
  Trend (last 5): ↓0.0107
  Total time: 6.3h
----------------------------------------------------------------------
  Loss                    Current       Best     Epoch1
  ──────────────────── ────────── ────────── ──────────
  lane_lines              -2.1166    -2.1166     8.5098
  lane_lines_prob          0.0965     0.0960     0.3796
  road_edges              -1.2402    -1.2402    18.5226
  lead                     1.3607     1.3490     3.5307
  lead_prob                0.2778     0.2686     0.3830
  pose                    -1.7808    -2.2325     0.4309
  road_transform          -7.0000    -7.0000    -2.4133
======================================================================
```

> **loss 为负值是正常的**：GaussianNLL 中包含 log(σ) 项，当模型学到合理的小 σ 值时（置信度高），log(σ) 为负，使总 loss 为负。这表示模型不仅预测准确（残差小），还给出了合理的不确定性估计。

### 早停（Early Stopping）

使用 `--early-stop N` 可在 val_loss 连续 N 个 epoch 不改善时自动终止训练：

```bash
# 15 个 epoch 无改善则停止
python tools/dashcam/train/train.py \
  --data-dirs data/training/carla_001 \
  --epochs 100 --early-stop 15
```

早停触发时的日志输出：
```
Epoch 42/100 ... | early_stop: best=-3.85, no_improve=15/15

Early stopping triggered at epoch 42 (no improvement for 15 epochs)
Best val_loss: -3.8500
```

每个 epoch 末尾会显示早停状态 `early_stop: best=<最佳val_loss>, no_improve=<等待次数>/<耐心值>`，方便观察收敛趋势。

### ONNX 导出

训练结束会自动导出 fp32 和 fp16 两个版本的 ONNX 模型。也可手动导出：

```bash
# 仅导出 fp32
python tools/dashcam/train/export_onnx.py \
  --checkpoint checkpoints/best.pt \
  --output driving_vision.onnx

# 同时导出 fp32 + fp16
python tools/dashcam/train/export_onnx.py \
  --checkpoint checkpoints/best.pt \
  --output driving_vision.onnx \
  --fp16
```

导出过程：
1. 加载 checkpoint，切换为 eval 模式
2. 执行 RepConv 重参数化（将训练时的 DWConv+BN+Identity 三分支融合为单个 DWConv）
3. 用 `torch.onnx.export` 导出 fp32 模型（opset 17，支持动态 batch）
4. 使用 onnxruntime 验证 PyTorch 与 ONNX 输出一致性
5. 若指定 `--fp16`，将 fp32 权重转换为 fp16，输入/输出保持 fp32 兼容（通过 Cast 节点自动转换）

**fp32 vs fp16 对比**：

| 版本 | 文件大小 | 精度 | 用途 |
|------|---------|------|------|
| fp32 | ~76MB | 完整精度 | 调试、评估、精度对比基准 |
| fp16 | ~38MB | 半精度 | 部署推理（体积减半，推理更快，精度损失极小） |

> openpilot 预训练的 `driving_vision.onnx` 使用 fp16 存储（~45MB）。fp16 转换仅影响权重存储精度，模型的输入和输出接口保持 fp32 不变，下游代码无需修改。

### 自训练模型推理

使用 `--custom-model` 可在 Carla 仿真中直接运行自训练的 ONNX 模型，无需 modeld 子进程、VisionIPC 或 cereal 消息传递：

```bash
# 启动 Carla
./tools/dashcam/start_carla.sh

# 使用自训练模型运行（fp32 或 fp16 均可）
python tools/dashcam/run.py \
  --perfect-cam --road-only --high-quality \
  --custom-model checkpoints/driving_vision.onnx

# 无头模式录制视频
python tools/dashcam/run.py \
  --perfect-cam --road-only --high-quality --no-display \
  --custom-model checkpoints/driving_vision_fp16.onnx \
  --save-video output.mp4 --max-frames 500
```

**工作原理**：

1. `--custom-model` 自动启用 `--road-only` 单摄像头模式
2. 跳过 Params/camerad/modeld/calibrationd/PubMaster/SubMaster 初始化
3. 每帧在主进程内执行完整推理管线（`infer.py:CustomModelInference`）：
   - `warp_image()` 将原始 RGB 透视变换到模型输入空间 (512×256)
   - `rgb_to_yuv420_6ch()` 转换为 6 通道 YUV420
   - 与上一帧拼接为 (12, 128, 256) 时序输入
   - onnxruntime 执行 ONNX 推理（GPU 优先，自动 fallback CPU）
   - MDN/BCE 解码 7 个输出张量
   - 构建 cereal modelV2 消息
4. 第一帧返回 None（需要两帧做时序配对），第二帧起正常渲染
5. visualizer 接收 modelV2 消息，渲染车道线、路边沿、前车检测

**输出解码**：

| 输出张量 | 原始维度 | reshape | 解码方式 | 结果 |
|---------|---------|---------|---------|------|
| `lane_lines` | 528 | (4, 33, 4) | MDN: μ=[:,:,:2], σ=exp([:,:,2:]) | μ(4,33,2), σ(4,33,2) |
| `lane_lines_prob` | 8 | (4, 2) | sigmoid, 取 [:,1] | prob(4,) |
| `road_edges` | 264 | (2, 33, 4) | MDN 同上 | μ(2,33,2), σ(2,33,2) |
| `lead` | 144 | (3, 6, 8) | MDN: μ=[:,:,:4], σ=exp([:,:,4:]) | μ(3,6,4), σ(3,6,4) |
| `lead_prob` | 3 | (3,) | sigmoid | prob(3,) |
| `pose` | 12 | (12,) | MDN: μ=[:6], σ=exp([6:]) | μ(6,), σ(6,) |
| `road_transform` | 12 | (12,) | MDN 同上 | μ(6,), σ(6,) |

---

## 完整工作流示例

从零开始：数据采集 → 训练 → 监控 → 导出模型。

```bash
# 0. 环境准备
source .venv/bin/activate
python -m ensurepip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 1. 启动 Carla
./tools/dashcam/start_carla.sh

# 2. 采集训练数据（20,000 帧，约 15 分钟）
python tools/dashcam/run.py \
  --perfect-cam --road-only \
  --record data/training/run_001 \
  --record-only --fast \
  --max-frames 20000

# 3. 检查采集的数据
python tools/dashcam/view_npz.py data/training/run_001/

# 4. 开始训练（RTX 4090 约 6 小时 / 100 epoch，早停可能提前结束）
python tools/dashcam/train/train.py \
  --data-dirs data/training/run_001 \
  --output-dir checkpoints \
  --epochs 100 --batch-size 16 --early-stop 15

# 5. 另开终端监控训练进度
python tools/dashcam/train/monitor.py checkpoints/training_log.csv --live

# 6. 训练完成后查看曲线和导出结果
python tools/dashcam/train/monitor.py checkpoints/training_log.csv --plot
ls -lh checkpoints/driving_vision*.onnx  # fp32 (~76MB) + fp16 (~38MB)

# 7. 用自训练模型在 Carla 中验证效果
python tools/dashcam/run.py \
  --perfect-cam --road-only --high-quality \
  --custom-model checkpoints/driving_vision.onnx
```

### 扩大数据规模

采集更多样化的数据可显著提升模型泛化能力：

```bash
# 不同地图
for town in Town04_Opt Town03_Opt Town06_Opt; do
  python tools/dashcam/run.py \
    --perfect-cam --road-only \
    --town $town \
    --record data/training/${town} \
    --record-only --fast \
    --max-frames 10000
done

# 或者使用批量脚本（多高度 × 多地图）
bash tools/dashcam/collect_multi_height.sh

# 全部数据一起训练
python tools/dashcam/train/train.py \
  --data-dirs data/training/Town04_Opt data/training/Town03_Opt data/training/Town06_Opt \
  --epochs 100
```
