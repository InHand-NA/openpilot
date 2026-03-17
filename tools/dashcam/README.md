# dashcam — 基于 Carla 仿真的感知数据采集与模型推理工具

dashcam 是一套将 openpilot 感知管线（modeld + calibrationd）接入 Carla 模拟器的端到端工具链，面向 LDW/FCW 行车预警系统研发。主要功能：

- **仿真环境**：Carla 0.9.16 双目相机 + 自车/NPC 自动驾驶
- **感知推理**：运行 openpilot 原版 modeld 或自训练模型，输出车道线/路沿/前车检测
- **在线标定**：自动估计相机安装角度与高度
- **数据采集**：双目 RGB + modeld 标签同步录制为 NPZ
- **模型训练**：FastViT 骨干 + MDN 损失的端到端训练框架

---

## 架构概览

```
┌───────────────────────────────────────────────────────────────┐
│                       主进程 run.py                            │
│                                                               │
│  carla_world.py          camerad.py         visualizer.py    │
│  ─────────────           ──────────         ──────────────    │
│  Carla 仿真              VisionIPC 服务端    感知结果渲染      │
│  - 双目 RGB 相机          road/wide 帧流     - 车道线 / 路沿   │
│  - 自车 + NPC                               - 前车标记        │
│  - 同步帧采集                                - 标定面板        │
│                                             - BEV 鸟瞰图      │
│  cereal 消息总线                                              │
│  发布: carState, deviceState, [liveCalibration]               │
│  订阅: modelV2, liveCalibration, [cameraOdometry]            │
└─────────────┬──────────────────────┬──────────────────────────┘
              │                      │
              v                      v
    ┌──────────────────┐   ┌──────────────────┐
    │  modeld 子进程    │   │ calibrationd 子进程│
    │  (标准 / 自定义)  │   │  （可选）         │
    │                  │   │                  │
    │  输入:            │   │  输入:            │
    │   VisionIPC 帧流  │   │   cameraOdometry │
    │   liveCalibration │   │   carState       │
    │                  │   │                  │
    │  输出:            │   │  输出:            │
    │   modelV2        │   │   liveCalibration│
    │   cameraOdometry │   │                  │
    └──────────────────┘   └──────────────────┘
```

**数据流**：Carla RGB → YUV 编码 → VisionIPC → modeld → modelV2 → 渲染/录制

---

## 文件结构

### 核心运行时

| 文件 | 说明 |
|------|------|
| `run.py` | 主入口：多进程编排、Carla 连接、消息发布/订阅、主循环 |
| `camerad.py` | VisionIPC 服务端，支持双目 / 单窄角 / 单广角三种模式 |
| `carla_world.py` | Carla 世界管理：自车/NPC 生成、双目相机挂载、同步帧采集 |
| `calibrationd.py` | 在线标定子进程：从 cameraOdometry 估计 pitch/yaw/height |
| `visualizer.py` | 感知渲染：车道线多边形、路沿、前车三角标记、BEV 面板 |
| `custom_modeld.py` | 自训练双目模型推理子进程（tinygrad TinyJit pkl，替代 modeld） |

### 数据采集与真值提取

| 文件 | 说明 |
|------|------|
| `dual_data_recorder.py` | 双目 NPZ 录制：road_rgb + wide_rgb + modeld 标签 |
| `modeld_label_extractor.py` | 从 modelV2/cameraOdometry 消息提取训练标签 |
| `lane_ground_truth.py` | Carla 地图 API 提取车道线/路沿几何真值 |
| `lead_ground_truth.py` | Carla 世界前车检测与时序轨迹真值 |
| `pose_ground_truth.py` | 从 Carla 帧差分计算自车 6DoF 运动真值 |

### 评估与可视化工具

| 文件 | 说明 |
|------|------|
| `lane_evaluator.py` | 车道线评估：模型输出 vs Carla GT，输出精度/召回等指标 |
| `eval_pretrained.py` | 预训练模型精度评估（MAE/RMSE 或逐帧可视化） |
| `view_dual_data.py` | 双目 NPZ 数据查看器：road/wide 并排 + 标签叠加 |
| `warp_example.py` | Warp 变换演示：原始 RGB → 模型输入空间 |

### Shell 脚本

| 文件 | 说明 |
|------|------|
| `start_carla.sh` | 启动 Carla 0.9.16 Docker 容器（NVIDIA GPU + Epic 画质） |
| `collect_multi_height.sh` | 批量采集：遍历多相机高度 × 多地图 |
| `sample.sh` | 快速采集示例脚本 |

### 训练框架 `train/`

| 文件 | 说明 |
|------|------|
| `train/config.py` | 模型与训练超参配置（ModelConfig / DualCameraModelConfig / TrainConfig） |
| `train/model.py` | FastViT 驾驶视觉模型（RepMixer 骨干 + 双路输出头） |
| `train/dataset.py` | 数据集：DualCameraDrivingDataset / CachedDualCameraDrivingDataset |
| `train/pretrained_model.py` | PretrainedVisionModel：onnx2torch 加载 openpilot 预训练权重，支持冻结/微调 |
| `train/preprocess_cache.py` | 预处理缓存生成：raw NPZ → warp+YUV 缓存（**26× 训练加速**） |
| `train/export_onnx.py` | PyTorch .pt → ONNX（RepConv 融合 + 输出切片元数据嵌入 + fp16） |
| `train/compile_tinygrad.py` | ONNX → tinygrad TinyJit pkl（供 custom_modeld.py 加载） |
| `train/export_pretrained_pt.py` | openpilot ONNX → PyTorch .pt（state_dict 或 TorchScript） |
| `train/export_pretrained_onnx.py` | PyTorch TorchScript .pt → ONNX（含 onnxsim 简化） |

### 数据清洗与划分 `train/`

| 文件 | 说明 |
|------|------|
| `train/clean_data.py` | T2 数据清洗：时序抽帧 + 置信度过滤 + PRE 帧检查，输出 `clean_log.txt` |
| `train/verify_clean.py` | T2 验证工具：3 项自动检查（抽帧/置信度/PRE帧） |
| `train/split_dataset.py` | T3.1 训练集划分：按 8:1:1 划分 train/val/test，排除 disabled 帧 |
| `train/h1_holdout_split.py` | T3.2 H1 退化验证集：从未选中帧中抽取 H1 holdout |
| `train/make_mini_dataset.py` | T3.3 Mini 数据集：提取小子集用于调试训练脚本 |
| `viz/inspect_clean_log.py` | 逐帧浏览 clean_log 中的帧及标注，支持 `d` 键禁用帧 |

---

## 前置条件

| 依赖 | 要求 |
|------|------|
| Carla | 0.9.16（Docker 镜像 `carlasim/carla:0.9.16`） |
| GPU | NVIDIA，CUDA（modeld 推理，推荐 RTX 4090） |
| Docker | 需要 `--runtime=nvidia` 支持 |
| Python | 3.11+，openpilot venv 已激活（`source .venv/bin/activate`） |
| PyTorch | 训练时需要，运行时不需要 |

---

## 快速开始

```bash
# 1. 激活环境
source .venv/bin/activate

# 2. 启动 Carla 服务端（首次自动拉取 Docker 镜像）
./tools/dashcam/start_carla.sh

# 3. 另开终端，运行 dashcam（使用 openpilot 原版 modeld）
python tools/dashcam/run.py --perfect-cam --high-quality
```

按 `q` 或 `ESC` 退出。

---

## 命令行参数

### 连接与场景

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | Carla 服务器地址 |
| `--port` | `2000` | Carla 服务器端口 |
| `--town` | `Town04_Opt` | Carla 地图名称 |
| `--spawn-point` | `16` | 自车出生点索引 |
| `--random-spawn` | — | 随机出生点（覆盖 `--spawn-point`） |
| `--num-npc` | `20` | NPC 车辆数量 |
| `--speed-range MIN MAX` | `20 140` | 自车目标速度范围（km/h） |

### 相机姿态

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera-pitch` | `5.0` | 相机俯仰角（度），正值向下 |
| `--camera-yaw` | `3.0` | 相机偏航角（度），正值向左 |
| `--camera-height` | `1.13` | 相机离路面高度（米） |
| `--perfect-cam` | — | 理想安装：pitch=0, yaw=0 |

### 标定模式

| 参数 | 说明 |
|------|------|
| （默认） | **在线标定模式**：启动 calibrationd 子进程，从视觉里程计实时估计 pitch/yaw/height |
| `--known-pose` | **已知姿态模式**：跳过 calibrationd，直接使用 `--camera-pitch`/`--camera-yaw` 作为固定标定值 |

### 运行控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--high-quality` | — | 高画质渲染（完整地图层 + 后处理） |
| `--no-display` | — | 无头模式，不显示窗口 |
| `--fast` | — | 全速运行，绕过 20 FPS 帧率限制 |
| `--save-video PATH` | `''` | 保存可视化为 MP4 文件 |
| `--max-frames N` | `0` | 最大帧数（0 = 无限） |
| `--wide-road-only` | — | 仅广角相机模式（modeld 使用 ecam intrinsics） |
| `--road-only` | — | 仅窄角相机模式（modeld 使用 fcam intrinsics） |
| `--height-comp` | — | 启用车高补偿（不推荐） |

### 数据采集

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--record-modeld DIR` | `''` | 双目采集：保存 road_rgb + wide_rgb + modeld 标签到指定目录，`--fast` 自动开启 |
| `--record-skip N` | `1` | 每 N 帧保存 1 帧（减少磁盘占用） |

### 自训练模型推理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--custom-modeld PATH` | `''` | 自训练模型路径（`.pkl` 直接加载，`.onnx` 自动编译为 tinygrad pkl） |

---

## 使用示例

```bash
# 标准运行（openpilot modeld，理想相机，高画质）
python tools/dashcam/run.py --perfect-cam --high-quality

# 在线标定（模拟相机装歪 5° 俯仰 + 3° 偏航，默认即在线标定）
python tools/dashcam/run.py --camera-pitch 5 --camera-yaw 3

# 已知姿态模式（跳过 calibrationd，直接使用指定角度）
python tools/dashcam/run.py --known-pose --camera-pitch 5 --camera-yaw 3

# 录制视频（500 帧后自动停止）
python tools/dashcam/run.py --perfect-cam --save-video output.mp4 --max-frames 500

# 仅广角相机模式
python tools/dashcam/run.py --perfect-cam --wide-road-only

# 双目数据采集（20000 帧，modeld 标注）
python tools/dashcam/run.py \
  --perfect-cam \
  --record-modeld data/dual_camera_train/Town04_001 \
  --max-frames 20000

# 使用自训练模型（ONNX 自动编译，首次慢）
python tools/dashcam/run.py \
  --perfect-cam --high-quality \
  --custom-modeld checkpoints/dual/driving_vision.onnx

# 使用已编译 pkl（快速启动）
python tools/dashcam/run.py \
  --perfect-cam --high-quality \
  --custom-modeld checkpoints/dual/driving_vision_tinygrad_cuda.pkl
```

---

## Carla 服务端

```bash
# 前台运行
./tools/dashcam/start_carla.sh

# 后台运行
DETACH=1 ./tools/dashcam/start_carla.sh

# 停止
docker kill $(docker ps -q --filter ancestor=carlasim/carla:0.9.16)
```

Carla 服务端监听 `localhost:2000`，使用 Epic 画质 + 离屏渲染。

---

## NPC 车辆管理

- NPC 使用 Carla Traffic Manager 自动驾驶
- **休眠机制**：距自车 > 150m 的 NPC 进入休眠
- **自动重生**：休眠 NPC 传送到自车 25~100m 范围内
- 保证自车周围始终有交通流量

---

## 在线标定

使用 `--online-calib` 时，calibrationd 子进程从 modeld 的 `cameraOdometry` 消息实时估计相机 pitch/yaw/height。

### 标定触发条件（需同时满足）

| 条件 | 阈值 |
|------|------|
| 车速 | > 15 mph |
| 视觉前向速度 | > 1.3 m/s |
| 偏航角速度 | < 2°/s（直线段） |

### 标定进度

每 100 个有效样本为一个 block，累积 **5 个 block** 后标定完成，最多维持 50 个 block 的滑动窗口。

### 标定状态

| 状态 | 说明 |
|------|------|
| `UNCALIBRATED` | 有效 block < 5，标定进行中 |
| `CALIBRATED` | 有效 block ≥ 5，pitch/yaw 在合法范围内 |
| `INVALID` | 有效 block ≥ 5，但 pitch/yaw 超出范围 |
| `RECALIBRATING` | 检测到安装变化，重新标定 |

---

## 可视化

渲染管线对齐 openpilot UI（`selfdrive/ui/onroad/model_renderer.py`）：

- **车道线**：绿色填充多边形，宽度 = `0.025 × prob` 米
- **路沿**：红色填充多边形，固定宽度 0.025 米
- **前车标记**：双层三角形（外层黄色 glow + 内层红色 chevron），大小随距离变化
- **BEV 面板**：鸟瞰图，80m × ±10m 范围，仅 `--custom-modeld` 模式下显示
- **标定面板**：实时显示 pitch/yaw/height/blocks/百分比

最终输出 2160×1080（原始 1928×1208 帧经 1.12× 缩放 + 居中裁剪）。

---

## 双目数据采集

### 采集流程

```
Carla 仿真（20 FPS）
  ↓
carla_world.py
  ├─ road_rgb (1928×1208)  — 窄角 RGB（FOV 40°）
  └─ wide_rgb (1928×1208)  — 广角 RGB（FOV 120°）
  ↓
VisionIPC → openpilot modeld
  ├─ modelV2       → ModeldLabelExtractor → 7 类感知标签
  └─ cameraOdometry → pose / road_transform / wide_from_device_euler
  ↓
DualCameraDataRecorder
  └─ 每帧保存压缩 NPZ（road_rgb + wide_rgb + 标签）
```

### 采集命令

```bash
# 启动 Carla
./tools/dashcam/start_carla.sh

# 采集 20000 帧双目数据
python tools/dashcam/run.py \
  --perfect-cam \
  --record-modeld data/dual_camera_train/Town04_001 \
  --max-frames 20000

# 跳帧采集（每 4 帧保存 1 帧，减少磁盘占用 4×）
python tools/dashcam/run.py \
  --perfect-cam \
  --record-modeld data/dual_camera_train/Town04_001 \
  --record-skip 4 --max-frames 20000

# 卡车高度（相机 2.4m）
python tools/dashcam/run.py \
  --perfect-cam --camera-height 2.4 \
  --record-modeld data/dual_camera_train/Town04_h2.4

# 切换地图
python tools/dashcam/run.py \
  --perfect-cam --town Town03_Opt \
  --record-modeld data/dual_camera_train/Town03_001
```

### 批量多高度采集

```bash
# 遍历高度 1.2/1.8/2.0/2.4/2.8m × 地图 Town04/03/06（共 15 组，30000 帧）
bash tools/dashcam/collect_multi_height.sh
```

### NPZ 数据格式（原始）

每个 `.npz` 文件包含单帧的完整图像与标注：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `road_rgb` | (1208, 1928, 3) | uint8 | 窄角原始 RGB |
| `wide_rgb` | (1208, 1928, 3) | uint8 | 广角原始 RGB |
| `lane_lines` | (4, 33, 3) | float32 | 4 条车道线 × 33 采样点 × (x, y, z) |
| `lane_lines_prob` | (4,) | float32 | 车道线存在概率 |
| `road_edges` | (2, 33, 3) | float32 | 2 条路沿 × 33 采样点 × (x, y, z) |
| `road_edges_prob` | (2,) | float32 | 路沿存在标志 |
| `lead` | (3, 6, 4) | float32 | 3 个前车候选 × 6 时间步 × (x, y, v, a) |
| `lead_prob` | (3,) | float32 | 前车存在概率 |
| `pose` | (6,) | float32 | 自车运动 [vx, vy, vz, wx, wy, wz] |
| `road_transform` | (6,) | float32 | 道路变换参数 |
| `wide_from_device_euler` | (3,) | float32 | 广角相机相对欧拉角 |
| `rpyCalib` | (3,) | float32 | 相机标定 [roll, −pitch, −yaw]（弧度） |
| `v_ego` | scalar | float32 | 自车速度（m/s） |
| `camera_height` | scalar | float32 | 相机高度（m） |
| `town` | str | — | Carla 地图名称 |
| `world_pose` | (6,) | float32 | Carla 世界坐标 [x,y,z,roll°,pitch°,yaw°] |
| `label_source` | str | — | `'modeld'` |

**坐标系**：校准坐标系（x=前向, y=右向, z=向下，重力对齐）。车道线在 X_IDXS（0~192m，33 点）处采样。

**车道线排列**：
```
[0] far-left   — 左相邻车道的左边界
[1] near-left  — 本车道左边界
[2] near-right — 本车道右边界
[3] far-right  — 右相邻车道的右边界
```

**前车数据含义**：
- 维度 0（3 个候选）：当前 / 2s 后 / 4s 后的最近前车
- 维度 1（6 个时间步）：[0, 2, 4, 6, 8, 10] 秒
- 维度 2（4 个参数）：前向距离 (m)、横向偏移 (m)、绝对速度 (m/s)、加速度 (m/s²)

### 数据查看

```bash
# 双目数据查看（← → 键翻页）
python tools/dashcam/view_dual_data.py data/dual_camera_train/Town04_001/

# Warp 变换效果演示
python tools/dashcam/warp_example.py data/dual_camera_train/Town04_001/000100.npz
```

----

## 迁移学习（预训练模型）

`train/pretrained_model.py` 提供 `PretrainedVisionModel` 类，用 `onnx2torch` 将 openpilot 的 `driving_vision.onnx`（23M 参数）包装为 PyTorch 模块，支持冻结骨干进行微调。

### ONNX 输出切片（1576 维）

openpilot 预训练 ONNX 的关键输出区段：

| 区段 | 切片 | 维度 | 说明 |
|------|------|------|------|
| `pose` | `[87:99]` | 12 | 自车运动 |
| `wide_from_device_euler` | `[99:105]` | 6 | 广角相机相对角度 |
| `road_transform` | `[105:117]` | 12 | 道路变换 |
| `lane_lines` | `[117:645]` | 528 | 车道线 MDN |
| `lane_lines_prob` | `[645:653]` | 8 | 车道线概率 |
| `road_edges` | `[653:917]` | 264 | 路沿 MDN |
| `lead` | `[917:1061]` | 144 | 前车 MDN |
| `lead_prob` | `[1061:1064]` | 3 | 前车概率 |

有效输出 977 维（跳过 meta、desire_pred、summarizer）。

### 预训练模型导出工具

```bash
# openpilot ONNX → PyTorch state_dict
python tools/dashcam/train/export_pretrained_pt.py

# openpilot ONNX → TorchScript（独立，无 onnx2torch 依赖）
python tools/dashcam/train/export_pretrained_pt.py --traced

# TorchScript .pt → ONNX（含 onnxsim 简化 + 元数据嵌入）
python tools/dashcam/train/export_pretrained_onnx.py
```

---

## 自训练模型推理

`--custom-modeld` 以独立子进程运行自训练双目模型，完全替代 openpilot modeld，使用相同的 VisionIPC 接口和 cereal 消息格式。

```bash
# ONNX 模式（首次自动编译为 tinygrad pkl，缓存到同目录）
python tools/dashcam/run.py \
  --perfect-cam --high-quality \
  --custom-modeld checkpoints/dual/driving_vision.onnx

# pkl 模式（直接加载已编译的 pkl，启动快）
python tools/dashcam/run.py \
  --perfect-cam --high-quality \
  --custom-modeld checkpoints/dual/driving_vision_tinygrad_cuda.pkl
```

**custom_modeld.py 推理管线**：
```
VisionIPC 帧 (road + wide)
  → DrivingModelFrame (OpenCL warp + loadyuv，与 openpilot modeld 完全一致)
  → CL Buffer → tinygrad Tensor
  → TinyJit 推理（GPU）
  → 解析输出切片 → MDN/BCE 解码（μ, σ, prob）
  → cereal modelV2 消息发布
```

---

## 数据清洗与划分（T2–T3）

多高度训练数据从采集到可训练的完整流水线。数据集目录（`dataset_dir`）可以是只读 NFS 挂载，所有输出写入独立的 `output_dir`。

### T2 — 数据清洗

对标注数据按规则筛选，输出 `clean_log.txt`（被选中的帧索引）。

**清洗规则**：
1. **时序抽帧**：每 5 个保存帧取 1 帧（save_every=4 × subsample=5 → step=20），等效 1 FPS
2. **最低置信度过滤**：L-inner prob ≤ 0.05 **且** R-inner prob ≤ 0.05 时丢弃
3. **PRE 帧检查**：帧 N 需要前一帧 N-4 存在（用于双帧输入拼接）
4. **H1~H6 同步**：仅对 H1 做分析，所有高度共享清洗结果

```bash
# 清洗（dataset_dir 只读，输出到 output_dir）
python tools/dashcam/train/clean_data.py \
    /nfs/openpilot-datasets/multi_height-0312/ \
    --output-dir data/multi_height-0312/ \
    --subsample 5 --min-ll-prob 0.05

# 仅统计，不写入文件
python tools/dashcam/train/clean_data.py \
    /nfs/openpilot-datasets/multi_height-0312/ \
    --dry-run

# 验证清洗结果（3 项检查）
python tools/dashcam/train/verify_clean.py \
    --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
    --output-dir data/multi_height-0312/
```

**输出**：`<output_dir>/clean_log.txt`，每行 `session_name/frame_number`。

### T3.1 — 训练集划分

将 clean_log 中的帧按 8:1:1 全局随机划分。自动排除 `disabled_frames.txt` 中被禁用的帧。

```bash
python tools/dashcam/train/split_dataset.py \
    --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
    --output-dir data/multi_height-0312/ \
    --ratio 0.8 0.1 0.1 --seed 42
```

**输出**：
- `train.txt` / `val.txt` / `test.txt` — 各子集帧索引
- `split_info.json` — 统计信息与 5 项验证结果

**内置验证**（全部 PASS 才正常退出）：
1. 总帧数守恒
2. 三子集无交集
3. 比例偏差 < 1%
4. Session 覆盖率 > 80%
5. 索引合法性（H1 标注文件存在）

### T3.2 — H1 退化验证集

从 clean_log **未选中**的 H1 帧中抽取 10%，用于训练时监测标准高度性能退化。

```bash
python tools/dashcam/train/h1_holdout_split.py \
    --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
    --output-dir data/multi_height-0312/ \
    --holdout-ratio 0.1 --seed 42
```

**输出**：
- `h1_holdout.txt` — holdout 帧索引
- `h1_holdout_info.json` — 统计信息与 5 项验证结果

### T3.3 — Mini 数据集

从完整划分中提取小子集，用于快速调试训练脚本。

```bash
python tools/dashcam/train/make_mini_dataset.py \
    --output-dir data/multi_height-0312/ \
    --mini-dir data/multi_height-0312-mini/ \
    --max-frames 100
```

**输出**：`mini_dir/` 下的 `train.txt`（100帧）、`val.txt`（12帧）、`test.txt`（12帧）、`h1_holdout.txt`（12帧）。

### 可视化浏览与帧禁用

逐帧浏览 clean_log 中的帧，查看 warp 后图像 + 标注叠加。

```bash
python tools/dashcam/viz/inspect_clean_log.py \
    --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
    --clean-log data/multi_height-0312/clean_log.txt \
    --height H1 --start 0
```

**快捷键**：

| 按键 | 功能 |
|------|------|
| ← → | 上一帧 / 下一帧 |
| PgUp / PgDn | ±10 帧 |
| Home / End | 首帧 / 末帧 |
| Tab | 切换窄焦 / 广角 |
| h | 切换高度 H1→H2→…→H6 |
| **d** | **禁用/启用当前帧**（实时保存 `disabled_frames.txt`） |
| l / e / v | 开关车道线 / 路沿 / 前车叠加 |
| b | 开关 BEV 鸟瞰面板 |
| i | 开关信息面板 |
| r | 切换原图 / warp 后图像 |
| s | 截图 |
| q / ESC | 退出 |

被禁用的帧会显示红色 **DISABLED** 横幅。重新运行 `split_dataset.py` 时自动排除这些帧。

### 端到端流水线示例

```bash
# 0. 采集（已完成）→ 标注（已完成）

# 1. 数据清洗
python tools/dashcam/train/clean_data.py \
    /nfs/openpilot-datasets/multi_height-0312/ \
    --output-dir data/multi_height-0312/

# 2. 可视化审查，按 'd' 标记问题帧
python tools/dashcam/viz/inspect_clean_log.py \
    --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
    --clean-log data/multi_height-0312/clean_log.txt

# 3. 训练集划分（自动排除 disabled 帧）
python tools/dashcam/train/split_dataset.py \
    --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
    --output-dir data/multi_height-0312/

# 4. H1 退化验证集
python tools/dashcam/train/h1_holdout_split.py \
    --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \
    --output-dir data/multi_height-0312/

# 5. Mini 数据集（调试用）
python tools/dashcam/train/make_mini_dataset.py \
    --output-dir data/multi_height-0312/ \
    --mini-dir data/multi_height-0312-mini/

# 6. 后续：预处理缓存（T4）→ 训练（T8）→ 评价（T9）
```

---

## 完整工作流

### 数据采集 → 训练 → 推理验证

```bash
# 0. 环境准备
source .venv/bin/activate
./tools/dashcam/start_carla.sh

# 1. 采集双目数据（约 15 分钟 / 20000 帧）
python tools/dashcam/run.py \
  --perfect-cam \
  --record-modeld data/dual_camera_train/Town04_001 \
  --max-frames 20000

# 2. 查看采集数据
python tools/dashcam/view_dual_data.py data/dual_camera_train/Town04_001/

# 3. 生成预处理缓存（~30 秒，32 核并行）
python tools/dashcam/train/preprocess_cache.py \
  data/dual_camera_train/Town04_001

# 4. 训练（见 train/ 目录，训练脚本正在基于迁移学习方案重建）

# 5. 从 checkpoint 导出 ONNX
python tools/dashcam/train/export_onnx.py \
  --checkpoint checkpoints/dual/best.pt \
  --output checkpoints/dual/driving_vision.onnx \
  --dual-camera --fp16

# 6. 编译 tinygrad pkl
DEV=CUDA python tools/dashcam/train/compile_tinygrad.py \
  checkpoints/dual/driving_vision.onnx

# 7. 用自训练模型验证效果
python tools/dashcam/run.py \
  --perfect-cam --high-quality \
  --custom-modeld checkpoints/dual/driving_vision_tinygrad_cuda.pkl
```

### 多高度 / 多地图数据扩充

```bash
# 自动遍历 5 个高度 × 3 个地图（30000 帧）
bash tools/dashcam/collect_multi_height.sh

# 手动多地图
for town in Town04_Opt Town03_Opt Town06_Opt; do
  python tools/dashcam/run.py \
    --perfect-cam --town $town \
    --record-modeld data/dual_camera_train/${town}_001 \
    --max-frames 10000
done
```
