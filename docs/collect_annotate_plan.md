# 多高度数据采集与标注工具技术文档

> 本文档描述 Carla 多高度同步数据采集 + 离线标注流水线的完整实现。
> 基于 [height_extension_design.md](height_extension_design.md) 第 4、5 节的设计，已全部完成开发。

---

## 目录

- [1. 总体数据流程](#1-总体数据流程)
- [2. 目录结构与数据格式](#2-目录结构与数据格式)
- [3. 采集与标注工具](#3-采集与标注工具)
  - [3.1 carla\_multi\_height\_world.py — Carla 多高度仿真环境](#31-carla_multi_height_worldpy--carla-多高度仿真环境)
  - [3.2 collect\_multi\_height.py — 单 session 采集](#32-collect_multi_heightpy--单-session-采集)
  - [3.3 run\_full\_collection.py — 批量采集编排器](#33-run_full_collectionpy--批量采集编排器)
  - [3.4 annotate\_multi\_height.py — 离线标注](#34-annotate_multi_heightpy--离线标注)
  - [3.5 annotate\_batch.py — 批量标注](#35-annotate_batchpy--批量标注)
- [4. 辅助模块](#4-辅助模块)
  - [4.1 modeld\_preprocess\_cl.py — GPU 预处理](#41-modeld_preprocess_clpy--gpu-预处理)
  - [4.2 lead\_ground\_truth.py — 前车真值提取](#42-lead_ground_truthpy--前车真值提取)
  - [4.3 pose\_ground\_truth.py — 位姿真值提取](#43-pose_ground_truthpy--位姿真值提取)
  - [4.4 modeld\_label\_extractor.py — 在线标签提取](#44-modeld_label_extractorpy--在线标签提取)
- [5. 可视化工具](#5-可视化工具)
  - [5.1 viz/inspect\_annotated.py — 单高度标注检查](#51-vizinspect_annotatedpy--单高度标注检查)
  - [5.2 viz/compare\_heights.py — 多高度对比](#52-vizcompare_heightspy--多高度对比)
  - [5.3 viz/label\_stats.py — 定量统计](#53-vizlabel_statspy--定量统计)
- [6. 端到端运行示例](#6-端到端运行示例)
- [7. 关键设计决策](#7-关键设计决策)

---

## 1. 总体数据流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Step A：Carla 多高度同步采集                                              │
│  run_full_collection.py → collect_multi_height.py                        │
│  + carla_multi_height_world.py                                           │
│                                                                          │
│  Carla 仿真（单次运行，6高度×2相机同步）                                     │
│    ├── H1/ road_000000.png, wide_000000.png, metadata.jsonl              │
│    ├── H2/ road_000000.png, wide_000000.png, metadata.jsonl              │
│    ├── ...                                                               │
│    └── H6/ road_000000.png, wide_000000.png, metadata.jsonl              │
│  clip_info.json：记录 pitch, yaw, heights, map, weather, save_every      │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Step B：离线标注（annotate_batch.py → annotate_multi_height.py）          │
│                                                                          │
│  H1/*.png (road + wide 双目图像)                                          │
│    → 从 clip_info.json 读取 pitch/yaw                                     │
│    → get_warp_matrix([0, pitch_rad, yaw_rad], K) 计算 warp 矩阵           │
│    → GPU OpenCL 预处理 → YUV420 (6,128,256) uint8                        │
│    → tinygrad TinyJit pkl 推理（prev + curr 双帧输入）                     │
│    → decode_model_output() 解码 MDN → canonical 标注                      │
│    → 质量过滤：lane_lines_prob[L-inn] > threshold AND [R-inn] > threshold │
│    → transform_annotation(h1→h_k) × 5 次：z += delta_h                   │
│    → 写入 annotations/{H1..H6}/{frame_id}.json                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Step C：预处理缓存（preprocess_cache.py）                                 │
│                                                                          │
│  annotations/{H_k}/{frame}.json + 源 session PNG 图像                     │
│    → 从 clip_info.json 读取 pitch/yaw                                     │
│    → get_warp_matrix → warp → YUV420 (6,128,256) uint8                   │
│    → 提取标注张量 + camera_height                                          │
│    → {H_k}_cache/{frame}.npz（road_yuv, wide_yuv, 标注张量）              │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Step D：模型训练（train.py）                                              │
│  输入：多个 {H_k}_cache/ 目录                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构与数据格式

### 2.1 采集输出目录结构

```
data/multi_height_0311/                    # 批量采集根目录
  Town04_ClearNoon_p5.0_y0.0/              # 单 session（map_weather_pitch_yaw）
    clip_info.json                          # session 元数据
    H1/                                     # 高度 1.22m
      road_000000.png                       # 窄焦相机 RGB (1928×1208)
      wide_000000.png                       # 广角相机 RGB (1928×1208)
      road_000004.png                       # save_every=4 时跳 4 tick
      wide_000004.png
      metadata.jsonl                        # 每帧一行 JSON 元数据
    H2/                                     # 高度 1.30m（同结构）
    ...
    H6/                                     # 高度 3.00m
  Town04_ClearNoon_p5.0_y0.0/annotations/  # 标注输出（与采集同级）
    clip_info.json                          # 复制自父 session + source_session_dir
    H1/
      000000.json                           # 标注 JSON（每帧一个）
      000004.json
    H2/
    ...
    H6/
  collection_progress.json                  # 采集进度（仅供参考）
  stats/                                    # label_stats.py 批量统计输出
    stats_summary.txt
    pitch_yaw_heatmap.png
    ...
```

### 2.2 clip_info.json 格式

```json
{
  "session_id": "Town04_ClearNoon_p5.0_y0.0",
  "map": "Town04",
  "weather": "ClearNoon",
  "camera": {
    "pitch_deg": 5.0,
    "yaw_deg": 0.0,
    "forward_offset_m": 0.8
  },
  "heights": {
    "H1": 1.22,
    "H2": 1.30,
    "H3": 1.50,
    "H4": 2.00,
    "H5": 2.50,
    "H6": 3.00
  },
  "simulation": {
    "fps": 20.0,
    "fixed_delta_seconds": 0.05,
    "num_npc": 40
  },
  "spawn": {
    "x": 123.4, "y": 456.7, "z": 0.5, "yaw": 45.3
  },
  "save_every": 4
}
```

> `pitch_deg` / `yaw_deg` 遵循 openpilot 约定（正值=nose-down/右偏），所有高度相机共用。
> 标注目录的 `clip_info.json` 额外包含 `source_session_dir` 字段，指向原始采集目录的相对路径。

### 2.3 metadata.jsonl 格式（每帧一行）

```json
{"frame": 0, "camera_height": 1.22, "v_ego": 15.5, "world_pose": [123.4, 456.7, 0.1, 0.0, 0.05, 45.3]}
```

### 2.4 标注 JSON 格式（canonical）

```json
{
  "frame_id": "000000",
  "camera_height": 1.22,
  "v_ego": 15.5,
  "world_pose": [123.4, 456.7, 0.1, 0.0, 0.05, 45.3],
  "label_source": "pretrained_h1",
  "ll_quality_pass": true,
  "lane_lines":             "...(4, 33, 3) [x, y_lat, z_height]",
  "lane_lines_prob":        "...(4,) sigmoid 后概率",
  "road_edges":             "...(2, 33, 3) [x, y_lat, z_height]",
  "road_edges_prob":        "...(2,) 全 1.0",
  "lead":                   "...(3, 6, 4) MDN means [x, y, v, a]",
  "lead_prob":              "...(3,) sigmoid 后概率",
  "pose":                   "...(6,) [trans(3), rot(3)] means",
  "road_transform":         "...(6,) [tx, ty, tz, 0, 0, 0] means",
  "wide_from_device_euler": "...(3,) [roll, pitch, yaw] means"
}
```

**标注字段说明：**

| 字段 | 形状 | 说明 |
|------|------|------|
| `lane_lines` | `(4, 33, 3)` float32 | 4 条车道线 × 33 距离点 × (x, y_lat, z_height)，x = X_IDXS |
| `lane_lines_prob` | `(4,)` float32 | 各车道线置信度 [L-outer, L-inner, R-inner, R-outer] |
| `road_edges` | `(2, 33, 3)` float32 | 2 条路沿 × 33 距离点 × (x, y_lat, z_height) |
| `road_edges_prob` | `(2,)` float32 | 全 1.0（模型无 per-edge 概率输出） |
| `lead` | `(3, 6, 4)` float32 | 3 个时间偏移(0/2/4s) × 6 未来时刻 × (x, y, v_abs, a) |
| `lead_prob` | `(3,)` float32 | 各前车预测的存在概率 |
| `pose` | `(6,)` float32 | [前进速度, 右偏速度, 下沉速度, roll角速度, pitch角速度, yaw角速度] |
| `road_transform` | `(6,)` float32 | [tx, ty, tz(=camera_height), 0, 0, 0] |
| `wide_from_device_euler` | `(3,)` float32 | 广角相机相对窄焦的欧拉角 [roll, pitch, yaw] |

---

## 3. 采集与标注工具

### 3.1 `carla_multi_height_world.py` — Carla 多高度仿真环境

**文件路径：** `tools/dashcam/carla_multi_height_world.py`

**职责：** 在单次 Carla 仿真中同时挂载 N×2（窄焦+广角）相机，按高度同步采集。

**核心类：**

```python
@dataclass
class CameraSlot:
    tag: str       # 'H1', 'H2', ..., 'H6'
    height: float  # 离地高度（米）

class MultiHeightCarlaWorld:
    def __init__(
        self,
        host='127.0.0.1', port=2000,
        town='Town04', weather='ClearNoon',
        spawn_point=16, random_spawn=False,
        camera_pitch_deg=5.0,            # openpilot 约定：正值=nose-down
        camera_yaw_deg=0.0,              # openpilot 约定：正值=右偏
        camera_forward_offset_m=0.8,     # 相机前向偏移
        camera_slots=None,               # 默认 [H1(1.22m), H6(3.0m)]
        num_npc=40,
        high_quality=False,
        speed_range=(40.0, 100.0),       # 目标速度范围 (km/h)
        speed_interval=(8.0, 20.0),      # 变速间隔 (秒)
    )
```

**关键参数：**

- 相机分辨率：`1928×1208`（openpilot 标准）
- 窄焦 FOV：40°，广角 FOV：120°
- 仿真帧率：20 FPS（`fixed_delta_seconds=0.05`）
- 角度约定转换：openpilot → Carla/UE4 左手系（`pitch=-pitch_deg, yaw=-yaw_deg`）

**帧同步设计：**

Carla 同步模式保证同一 `world.tick()` 的所有 sensor 共享相同物理状态和 `image.frame` 帧号。但 sensor 回调通过独立流线程异步送达，`tick()` 返回时不保证全部到达。

```python
def get_frames(self) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """仅当所有相机（N高度×2）帧号完全一致时返回，否则返回 None。"""
    with self._lock:
        entries = [self._latest[slot.tag][key]
                   for slot in self._camera_slots for key in ('road', 'wide')]
        if any(e is None for e in entries):
            return None  # 尚有相机未收到数据
        frame_ids = [e[0] for e in entries]
        if len(set(frame_ids)) != 1:
            return None  # 帧号不一致，等待滞后回调
        return {slot.tag: (road_rgb.copy(), wide_rgb.copy()) for ...}
```

**天气预设支持：** ClearNoon, ClearSunset, CloudyNoon, WetNoon, WetSunset, MidRainSunset, SoftRainNoon

**关闭顺序：** 禁用 autopilot → 停止 sensor listener → 禁用 TM 同步 → 最后 tick → 批量 destroy actors → 切换 async 模式

---

### 3.2 `collect_multi_height.py` — 单 session 采集

**文件路径：** `tools/dashcam/collect_multi_height.py`

**职责：** 运行单个 Carla session，多高度同步采集原始 RGB 帧（PNG）+ 元数据（JSONL）。采集阶段不运行模型推理。

**命令行接口：**

```bash
python tools/dashcam/collect_multi_height.py \
    --phase quick                 # quick | full | custom
    --output-base data/multi_height
    --max-frames 20000            # 每 session 采集帧数
    --save-every 4                # 每 N tick 保存一帧（1或4）
    [--heights H1 H6]             # custom 模式
    [--map Town04]                # custom 模式
    [--weather ClearNoon]         # custom 模式
    [--pitch 5.0] [--yaw 0.0]    # custom 模式
    [--num-npc 40]
    [--random-spawn]              # 随机出生点
    [--no-display]
    [--speed-range 40.0 100.0]
    [--speed-interval 8.0 20.0]
```

**Phase 配置：**

| phase | heights | scenes | pitch×yaw | max-frames |
|-------|---------|--------|-----------|------------|
| `quick` | H1, H6 | Town04/ClearNoon | (5°, 0°) | 20000 |
| `full` | H1~H6 | 6 种 map×weather | 34 种 pitch×yaw | 8000 |
| `custom` | 参数指定 | 参数指定 | 参数指定 | 参数指定 |

**`save_every` 机制：**
- `save_every=4`：每 4 个 tick 保存一帧（等效 5 FPS），采集速度提升 4×
- 帧文件名按 `abs_tick` 编号：`road_000000.png`, `road_000004.png`, `road_000008.png`, ...
- 约束：`save_every` 必须为 1 或 4（与 `TEMPORAL_SKIP=4` 对齐）

**核心函数 `collect_session()`：**

```python
def collect_session(
    output_base, session_tag, heights, map_name, weather,
    pitch_deg, yaw_deg, max_frames, host, port, num_npc,
    spawn_point, random_spawn, high_quality, display,
    speed_range, speed_interval, save_every,
) -> Path:
    """采集单个 session，返回 session 目录路径。支持断点续采。"""
```

- 创建高度子目录，初始化 `MultiHeightCarlaWorld`
- 保存 `clip_info.json`（含 `save_every` 字段）
- 10 tick 热身等待车辆物理稳定
- 主循环：`tick()` → `get_frames()`（轮询直到帧号一致）→ 按 `save_every` 间隔写 PNG + JSONL
- 异步 PNG 写入（`ThreadPoolExecutor`），每 100 帧 drain futures
- 支持断点续采：读取已有帧数，从 `len(existing) * save_every` tick 继续
- OpenCV 预览窗口（可选）

---

### 3.3 `run_full_collection.py` — 批量采集编排器

**文件路径：** `tools/dashcam/run_full_collection.py`

**职责：** 编排多 scene × 多 pitch-yaw 的批量采集，支持自动重试、断点续跑、进度追踪。

**命令行接口：**

```bash
python tools/dashcam/run_full_collection.py \
    --output-base data/multi_height     # 输出根目录
    --max-frames 120                    # 每 session 帧数（名义中心 2×）
    --num-npc 40                        # NPC 数量
    --save-every 4                      # 每 N tick 保存
    --max-retries 3                     # 每 session 最大重试次数
    --retry-delay 30                    # 重试间隔（秒）
    --start-from 0                      # 跳过前 N 个 session
    --speed-range 40.0 100.0
    --speed-interval 8.0 20.0
    [--list]                            # 仅列出 session 计划
    [--no-display]
```

**场景矩阵：**

```
SCENES = [
    ('Town03', 'ClearNoon'),    ('Town03', 'WetSunset'),
    ('Town05', 'ClearSunset'),  ('Town05', 'CloudyNoon'),
    ('Town06', 'ClearNoon'),    ('Town06', 'ClearSunset'),
    ('Town06', 'CloudySunset'), ('Town06', 'WetNoon'),
]
```

**Pitch-Yaw 姿态矩阵（33 个采样点）：**

```
pitch \ yaw  | -3°  | -1.5° | 0°  | +1.5° | +3°
-------------|------|-------|-----|-------|------
  -1.5°      |  ○   |       |  ○  |       |  ○     3 个
   0.0°      |  ○   |       |  ○  |       |  ○     3 个
  +1.5°      |      |  ○    |  ○  |  ○    |        3 个
  +3.0°      |  ○   |  ○    |  ○  |  ○    |  ○     5 个
  +4.0°      |  ○   |  ○    |  ○  |  ○    |  ○     5 个
  +5.0°(名义)|  ○   |  ○    |  ●  |  ○    |  ○     5+1 (中心2×)
  +6.0°      |  ○   |  ○    |  ○  |  ○    |  ○     5 个
  +7.0°      |      |  ○    |  ○  |  ○    |        3 个
```

`●` = 名义中心 (5°, 0°)，采集 `2 × max_frames`。

**总 session 数：** 8 scenes × 33 poses = 264 sessions

**断点续跑机制：**
- 跳过已完成的 session（检查 H1 目录中的 PNG 帧数）
- 部分采集的 session 自动从已有帧数处继续
- `collection_progress.json`：纯信息性，不影响续跑逻辑
- 优雅 Ctrl+C：完成当前 session 清理后停止；再次 Ctrl+C 强制退出

---

### 3.4 `annotate_multi_height.py` — 离线标注

**文件路径：** `tools/dashcam/annotate_multi_height.py`

**职责：** 对采集的 H1 图像运行 tinygrad 推理，解码标注，变换到所有高度，输出 JSON 标注。

**命令行接口：**

```bash
python tools/dashcam/annotate_multi_height.py \
    data/multi_height_0311/Town04_ClearNoon_p5.0_y0.0/ \
    --onnx selfdrive/modeld/models/driving_vision.onnx \
    --output .../annotations/              # 默认 session_dir/annotations/
    --heights H1 H6                        # 指定高度（默认全部）
    --min-ll-prob 0.1                      # 质量过滤阈值
    --no-gpu-preprocess                    # 禁用 GPU OpenCL，回退 CPU
```

**推理引擎：** tinygrad TinyJit（非 PyTorch/onnx2torch）
- 首次运行自动编译 ONNX → `driving_vision_tinygrad_CUDA.pkl`
- 同时生成 `driving_vision_metadata.pkl`（含 input_shapes、output_slices）
- 后续运行直接加载 pkl，跳过编译

**模型输入：**
- 当前帧 + 前一帧的 YUV420 拼接 → `(1, 12, 128, 256)` uint8
- 时序间隔 `TEMPORAL_SKIP=4`：当前帧与前第 4 个 tick 的帧配对
- `save_every=4` 时 buffer 深度=2（相邻两帧刚好间隔 4 tick）
- `save_every=1` 时 buffer 深度=5（跳 4 帧取 prev）

**处理流程：**

```
1. 加载 clip_info.json → pitch/yaw → rpyCalib = [0, pitch_rad, yaw_rad]
2. 计算 warp 矩阵（narrow + wide）
3. 加载/编译 tinygrad 模型
4. 逐帧处理 H1 图像：
   a. GPU OpenCL 预处理：BGR → NV12 → warp → loadyuv → (6,128,256) uint8
   b. 拼接 [prev, curr] → (1, 12, 128, 256)
   c. tinygrad 推理 → 解码 MDN 输出
   d. 质量过滤：ll_prob[1] > threshold AND ll_prob[2] > threshold
   e. 对每个高度 h_k：transform_annotation(h1→h_k) + 写入 JSON
```

**`decode_model_output()` — MDN 解码：**

遵循 openpilot 非交错格式 `[all_means | all_log_sigma]`：

| 模型输出 | 原始维度 | 解码方式 | canonical 输出 |
|----------|---------|---------|---------------|
| lane_lines | (528,) | 前 264 = means → reshape(4,33,2) [y,z] + 拼 x | (4,33,3) |
| lane_lines_prob | (8,) | reshape(4,2) → sigmoid([:,1]) | (4,) |
| road_edges | (264,) | 前 132 = means → reshape(2,33,2) [y,z] + 拼 x | (2,33,3) |
| lead | (144,) | 前 72 = means → reshape(3,6,4) [x,y,v,a] | (3,6,4) |
| lead_prob | (3,) | sigmoid | (3,) |
| pose | (12,) | 前 6 = means [trans(3), rot(3)] | (6,) |
| road_transform | (12,) | 前 6 = means | (6,) |
| wide_from_device_euler | (6,) | 前 3 = means [roll, pitch, yaw] | (3,) |

**`transform_annotation()` — 高度变换：**

```python
def transform_annotation(canonical, h1, h_k):
    delta_h = h_k - h1
    out = {k: v.copy() for k, v in canonical.items()}
    out['lane_lines'][:, :, 2] += delta_h      # z_height 列
    out['road_edges'][:, :, 2] += delta_h      # z_height 列
    out['road_transform'][2]   += delta_h      # tz 分量
    return out
```

仅修正 z 分量（相机高度差的线性偏移），近场近似在 X > ~10m 有效。

**质量过滤规则：**
```
PASS = lane_lines_prob[1] > min_ll_prob AND lane_lines_prob[2] > min_ll_prob
```
即 L-inner 和 R-inner 两条车道线置信度均超过阈值。

---

### 3.5 `annotate_batch.py` — 批量标注

**文件路径：** `tools/dashcam/annotate_batch.py`

**职责：** 扫描根目录下所有 session 子目录，逐一调用 `annotate_session()` 进行标注。模型仅加载一次，GPU 预处理器跨 session 复用。

**命令行接口：**

```bash
python tools/dashcam/annotate_batch.py data/multi_height_0311/ \
    --onnx selfdrive/modeld/models/driving_vision.onnx \
    --heights H1 H6                    # 可选：仅标注指定高度
    --min-ll-prob 0.1                  # 质量过滤阈值
    --dry-run                          # 仅预览，不实际标注
    --force                            # 强制重新标注
    --no-gpu-preprocess
```

**断点续跑：** 已完成标注的 session（annotations/ 中帧数 ≥ 源帧数）自动跳过。`--force` 可强制重新标注。

**环境变量：** `DEV`（默认 `CUDA`），`PYOPENCL_CTX`（默认空）

---

## 4. 辅助模块

### 4.1 `modeld_preprocess_cl.py` — GPU 预处理

**文件路径：** `tools/dashcam/modeld_preprocess_cl.py`（类名 `ModeldInputPreprocessorCL`）

精确复制 openpilot modeld 的三阶段 GPU 预处理流水线：

```
Stage 1 (rgb_to_nv12.cl):  BGR (1928×1208×3) → NV12 (Y + UV)
Stage 2 (transform.cl):    NV12 + warp_matrix → warped Y(512×256) + U(256×128) + V(256×128)
Stage 3 (loadyuv.cl):      Y/U/V → 6-channel [y0, y1, y2, y3, U, V] (6×128×256) uint8
```

```python
with ModeldInputPreprocessorCL() as pp:
    yuv = pp.process(bgr, warp_matrix)  # (6, 128, 256) uint8
```

### 4.2 `lead_ground_truth.py` — 前车真值提取

**文件路径：** `tools/dashcam/lead_ground_truth.py`（类名 `LeadGroundTruth`）

从 Carla world actors 中提取前方车辆的 3D 位置、速度、加速度，转为 openpilot lead 格式。

**输出格式：** `lead_data (3,6,4)` + `lead_prob (3,)` — 与模型输出完全一致

**过滤条件：**
- 前向距离：2.0 ~ 200.0m
- 横向距离：< 2.0m（同车道宽度）
- 高度：-1.0 ~ 5.0m（排除天桥）
- 加速度 clip：-10 ~ 5 m/s²

**三个 lead 选择对应时间偏移 0s / 2s / 4s（非三辆车）。**

### 4.3 `pose_ground_truth.py` — 位姿真值提取

**文件路径：** `tools/dashcam/pose_ground_truth.py`（类名 `PoseGroundTruth`）

从连续 Carla 帧计算：
- `pose (6,)`：平移速度(m/s) + 角速度(rad/s)，calibrated frame
- `road_transform (6,)`：`[0, 0, camera_height, 0, 0, 0]`
- `wide_from_device_euler (3,)`：广角相机相对窄焦的固定欧拉角

### 4.4 `modeld_label_extractor.py` — 在线标签提取

**文件路径：** `tools/dashcam/modeld_label_extractor.py`（类名 `ModeldLabelExtractor`）

从 cereal `modelV2` + `cameraOdometry` 消息提取标签，用于在线采集（`run.py` 流水线）。输出格式与离线标注完全一致。

---

## 5. 可视化工具

所有可视化工具位于 `tools/dashcam/viz/` 目录。

### 5.1 `viz/inspect_annotated.py` — 单高度标注检查

**检查点：** Step B 完成后

**目的：** 在 warp 后的模型输入图像（1024×512 显示）上叠加标注，验证推理质量和变换正确性。

```bash
python tools/dashcam/viz/inspect_annotated.py \
    data/.../annotations/ \
    --height H1              # 初始高度
    --start 0                # 起始帧
    --filter-low             # 仅显示低置信度帧
    --min-ll-prob 0.5        # 过滤阈值
```

**显示布局：**
```
┌─────────────────────────────────────┬──────────┐
│  Warp 后 road 图像 (1024×512)      │  BEV     │
│  · 车道线多边形（绿色，alpha=prob） │  俯视图  │
│  · 路沿线（红色）                   │  300×*   │
│  · 前车三角形 + 距离标注            │          │
│  · 信息面板（左上，可选）           │          │
├─────────────────────────────────────┼──────────┤
│  概率条 [L-out] [L-inn] [R-inn] [R-out]       │
└────────────────────────────────────────────────┘
```

**键盘操作：**

| 键 | 功能 | 键 | 功能 |
|----|------|----|------|
| `←/→` | 前/后帧 | `Tab` | 窄焦/广角切换 |
| `PgUp/PgDn` | ±10 帧 | `h` | 循环切换高度 |
| `Home/End` | 首/末帧 | `l/e/v` | 切换 车道线/路沿/前车 |
| `b` | BEV 面板 | `i` | 信息面板 |
| `r` | 原始/warp 图像 | `f` | 仅显示低质量帧 |
| `s` | 截图 | `q/ESC` | 退出 |

---

### 5.2 `viz/compare_heights.py` — 多高度对比

**检查点：** Step B 完成后

**目的：** 同一帧并排展示 H1~H6 的 warp 后图像 + 标注，直观对比多高度感知和 `transform_annotation` 正确性。

```bash
python tools/dashcam/viz/compare_heights.py \
    data/.../annotations/ \
    --heights H1 H4 H6       # 可指定子集
    --start 0
    --no-annotations          # 仅原始 warp 图像
    --min-ll-prob 0.5
```

**两种显示模式：**

- **网格模式（默认）：** 2×3 面板（H1~H6），每面板 512×256
- **全幅模式：** 单高度 1024×512，带 BEV 侧栏

**额外键盘操作：** `g` 切换网格/全幅 | `1-6` 选择高度 | `a` 标注开关 | `z` 显示 z_height 数值

**核心验证点：**
1. 所有高度地平线位置一致（共享 pitch）
2. z_height 偏移 = H_k - H1（按 `z` 查看数值）
3. H5/H6 近场 X<10m 区域空白（近场盲区）
4. 所有高度车道线 y_lat 一致，仅 z 不同

---

### 5.3 `viz/label_stats.py` — 定量统计

**检查点：** Step B 完成后，定量质量审计

**目的：** 生成统计报告和分布图，支持单 session 和批量两种模式（自动检测）。

```bash
# 单 session 模式
python tools/dashcam/viz/label_stats.py data/.../annotations/

# 批量模式（自动扫描子目录）
python tools/dashcam/viz/label_stats.py data/multi_height_0311/ \
    --output data/multi_height_0311/stats/ \
    --min-ll-prob 0.5 \
    --sample 0              # 0=全部帧
```

**生成内容（`stats/` 目录）：**

| 文件 | 说明 |
|------|------|
| `stats_summary.txt` | 纯文本摘要报告 |
| `lane_prob_distribution.png` | 各高度车道线置信度分布直方图 |
| `filter_rates.png` | 各高度过滤通过率柱状图 |
| `z_height_transform.png` | z_height 变换一致性验证（各高度曲线应平行，间距=ΔH） |
| `lead_detection_rate.png` | 各高度前车检测率 |
| `conf_dist_lane_lines_prob.png` | 车道线各分量置信度分布 |
| `conf_dist_road_edges_prob.png` | 路沿置信度分布 |
| `conf_dist_lead_prob.png` | 前车置信度分布 |
| `conf_dist_total.png` | 全数据集置信度汇总 |
| `session_pass_rates.png` | （批量）各 session 通过率排序 |
| `pitch_yaw_heatmap.png` | （批量）pitch×yaw 通过率热力图 |

**文本报告内容（批量模式）：**

```
=== Batch Multi-Height Annotation Quality Summary ===
Data root: multi_height_0311
Sessions: 128

--- Aggregate per height (all sessions) ---
Height       Total    Pass  PassRate    L0 Med    R0 Med   Lead%
H1 1.22m   15840   11689   73.8%      0.90      0.88   11.8%
...

--- z_height transform consistency: [OK] PASS ---
  H2: expected dH=0.08m  measured=0.080m  error=0.0000m [OK]
  ...

--- Pass rate by pitch x yaw ---
  pitch    yaw  Sessions   Frames   Pass%   Lead%
  +3.0   +0.0         4      480   99.2%    1.0%
  ...

--- Confidence Metric Distribution (per height + total) ---
  [lane_lines_prob]
  Group    Component    Min     Max    Mean  Median   Count
  H1       L-inner    0.001   0.997   0.681   0.895   15840
  ...
```

**统计指标定义：**
- **pass_rate**：L-inner 和 R-inner 置信度均 > min_ll_prob 的帧占比
- **lead_rate**：至少 1 个 lead_prob > 0.3 的帧占比
- **z_height 一致性**：各高度间 z 偏移量与预期 ΔH 的误差（<0.1m 为 PASS）

---

## 6. 端到端运行示例

### 6.1 快速验证（单 session）

```bash
# ── Step A：采集 ─────────────────────────────────────────────
# 需要 Carla 服务器运行
python tools/dashcam/collect_multi_height.py \
    --phase custom \
    --heights H1 H6 \
    --map Town04 --weather ClearNoon \
    --pitch 5.0 --yaw 0.0 \
    --max-frames 120 --save-every 4 \
    --random-spawn --no-display

# ── Step B：标注 ─────────────────────────────────────────────
DEV=CUDA python tools/dashcam/annotate_multi_height.py \
    data/multi_height/Town04_ClearNoon_p5.0_y0.0/ \
    --min-ll-prob 0.3

# ── 人工检查 ─────────────────────────────────────────────────
# 查看 H1 标注质量
python tools/dashcam/viz/inspect_annotated.py \
    data/multi_height/Town04_ClearNoon_p5.0_y0.0/annotations/

# 多高度对比
python tools/dashcam/viz/compare_heights.py \
    data/multi_height/Town04_ClearNoon_p5.0_y0.0/annotations/

# 统计报告
python tools/dashcam/viz/label_stats.py \
    data/multi_height/Town04_ClearNoon_p5.0_y0.0/annotations/
```

### 6.2 批量训练数据采集

```bash
# ── Step A：批量采集（264 sessions，支持断点续跑）─────────────
python tools/dashcam/run_full_collection.py \
    --output-base data/multi_height_0311 \
    --max-frames 120 --save-every 4 \
    --num-npc 40 --no-display

# Carla 崩溃后直接重新运行，自动跳过已完成的 session
python tools/dashcam/run_full_collection.py \
    --output-base data/multi_height_0311 \
    --max-frames 120 --save-every 4

# ── Step B：批量标注 ─────────────────────────────────────────
# 预览
python tools/dashcam/annotate_batch.py data/multi_height_0311/ --dry-run

# 执行标注
DEV=CUDA python tools/dashcam/annotate_batch.py data/multi_height_0311/ \
    --min-ll-prob 0.1

# ── 质量审计 ─────────────────────────────────────────────────
python tools/dashcam/viz/label_stats.py data/multi_height_0311/ \
    --min-ll-prob 0.3

# ── Step C：预处理缓存 ───────────────────────────────────────
# 对每个 session 的各高度目录执行
python tools/dashcam/train/preprocess_cache.py \
    data/multi_height_0311/Town04_ClearNoon_p5.0_y0.0/annotations/H1

# ── Step D：训练 ─────────────────────────────────────────────
python tools/dashcam/train/train.py \
    --cache-dirs data/.../H1_cache data/.../H6_cache \
    --output-dir checkpoints/multi_height_v1 \
    --epochs 100 --batch-size 16 --early-stop 20
```

---

## 7. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| **存储格式** | PNG + JSON（非 NPZ） | 图像和标注解耦；PNG 便于浏览和调试；JSON 人类可读 |
| **推理引擎** | tinygrad TinyJit pkl | 与 openpilot 推理路径一致（非 PyTorch/onnx2torch）；自动编译缓存 |
| **GPU 预处理** | OpenCL (modeld_preprocess_cl.py) | 精确复制 openpilot modeld 三阶段流水线（rgb_to_nv12 → transform → loadyuv） |
| **Warp 参数来源** | 方案A：从 clip_info.json 读取固定 pitch/yaw | Carla 精确已知相机姿态，无需 rpyCalib 标定收敛 |
| **标注生成时机** | 离线批量处理（非在线采集时） | 简化采集流程；GPU 批量推理更高效；模型可升级无需重新采集 |
| **标注格式** | canonical JSON（非 flat tensor） | 与 extract_targets() 兼容；可视化工具直接使用；人类可读 |
| **多高度同步** | 同一 session 多相机 + frame_id 全匹配 | 零额外仿真成本；时序完美对齐；H1 标注精确对应每帧场景 |
| **save_every=4** | 每 4 tick 保存一帧（5 FPS） | 采集速度提升 4×；与 TEMPORAL_SKIP=4 对齐，标注时 buffer 深度=2 |
| **断点续跑** | 基于文件计数（非 progress JSON） | 鲁棒：progress JSON 仅供参考，实际基于 H1 目录帧数判断 |
| **质量过滤** | L-inner & R-inner 置信度双阈值 | 只关心行车主视野的内侧车道线，外侧和路沿不参与过滤判定 |
| **z 高度变换** | 线性偏移 z += delta_h | 近场近似，X > ~10m 有效；远场误差可忽略（∝ ΔH/X → 0） |
| **随机出生点** | random_spawn=True | 增加场景多样性，避免同一 spawn_point 的数据偏差 |
| **NPC 密度** | 默认 40 辆 | 平衡仿真性能和场景真实性；lead 检测率约 12% 符合预期 |
