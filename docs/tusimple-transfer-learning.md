# 多高度单目相机 TuSimple 数据采集与标注系统 — 实现方案

## Context

项目需要一套基于 Carla 仿真的数据采集和标注系统，用于生成 TuSimple 格式的车道线检测训练数据。实际项目使用一款 1920×1080、HFOV 120° 的鱼眼相机，安装在不同高度的车辆上（轿车/SUV/卡车）。系统利用 openpilot 预训练模型在 3D 空间进行车道线标注，然后投影到各单目相机的 2D 图像平面。

## 相机编号约定

| 编号 | 类型 | 分辨率 | FOV | 高度 | 用途 |
|------|------|--------|-----|------|------|
| **H0** | openpilot 参考双目 | 1928×1208 | narrow 40° / wide 120° | 1.22m | modeld 3D 标注基准 |
| **H1** | Mono 单目 | 1920×1080 | 120° | 1.22m | TuSimple 训练数据 |
| **H2** | Mono 单目 | 1920×1080 | 120° | 1.30m | TuSimple 训练数据 |
| **H3** | Mono 单目 | 1920×1080 | 120° | 1.50m | TuSimple 训练数据 |
| **H4** | Mono 单目 | 1920×1080 | 120° | 2.00m | TuSimple 训练数据 |
| **H5** | Mono 单目 | 1920×1080 | 120° | 2.50m | TuSimple 训练数据 |
| **H6** | Mono 单目 | 1920×1080 | 120° | 3.00m | TuSimple 训练数据 |

- **H0** 是 openpilot 标准双目相机（narrow + wide），仅安装在 1.22m 高度，专用于 modeld 推理获取 3D 标注
- **H1~H6** 是实际项目相机的仿真（1920×1080, FOV=120° 针孔），安装在 6 个不同高度
- H0 与 H1 安装在相同高度 (1.22m)，但内参和分辨率不同

## 系统架构总览

系统分为四个独立阶段，各自有对应的脚本，可独立运行：

```
┌──────────────┐     ┌───────────────┐     ┌────────────────────┐     ┌────────────────────┐
│   Phase 1    │     │    Phase 2    │     │      Phase 3       │     │      Phase 4       │
│   数据采集    │ ──→ │    3D 标注    │ ──→ │   数据清洗与抽样    │ ──→ │      2D 投影       │
│  collect.py  │     │annotate_3d.py │     │clean_and_sample.py │     │project_tusimple.py │
└──────┬───────┘     └───────┬───────┘     └─────────┬──────────┘     └─────────┬──────────┘
       ▼                     ▼                       ▼                          ▼
  原始图像 + 元数据     3D 车道线 JSON          帧列表 (splits)          TuSimple JSON + JPEG
```

### Phase 1: 数据采集 (`collect.py`)

```
  ┌─────────────────────────────────────────────────────────┐
  │                   Carla 仿真环境                          │
  │                                                         │
  │   ┌──────────────────┐     ┌─────────────────────────┐  │
  │   │ H0 参考双目       │     │  H1~H6 Mono 单目 × 6   │  │
  │   │ narrow 1928×1208 │     │  1920×1080, FOV=120°    │  │
  │   │ wide   1928×1208 │     │  pinhole 针孔模型        │  │
  │   │ height=1.22m     │     │  height=1.22~3.00m      │  │
  │   └──────────┬───────┘     └────────────┬────────────┘  │
  └──────────────┼──────────────────────────┼───────────────┘
                 ▼                          ▼
           H0/road_*.png              H1/ ~ H6/
           H0/wide_*.png              *.png + metadata
           H0/metadata.jsonl
```

### Phase 2: 3D 标注 (`annotate_3d.py`)

```
  H0 图像 ──→ modeld 推理 ──→ 3D 车道线 (4,33,3) ──→ H1 3D labels
                                    │
                                    │ z += Δh (高度变换)
                                    ▼
                              H2~H6 3D labels
```

### Phase 3: 数据清洗与抽样 (`clean_and_sample.py`)

```
  3D labels ──→ 质量过滤 ──→ 时间抽样 ──→ 训练/验证/测试划分
                  │              │                │
                  │ ll_prob >    │ 每 N 帧取 1    │ 8:1:1 随机
                  │ threshold    │ (去时序冗余)    │
                  ▼              ▼                ▼
              quality_pass   sampled_frames   train.txt
              log.txt        log.txt          val.txt
                                              test.txt
```

### Phase 4: 2D 投影 (`project_tusimple.py`)

```
  帧列表 (Phase 3) ──→ 仅处理选中帧
  H1~H6 3D labels  ──→  K_crop 投影     ──→  1280×720 TuSimple
  H1~H6 Mono 图像  ──→  ROI 裁剪(~70°)  ──→  images/*.jpg
                        + resize              labels.json
```

## 一、鱼眼相机 Carla 仿真方案

### 问题

实际项目使用 HFOV 120° 鱼眼相机。Carla 只支持标准针孔 (pinhole) 相机模型。

### 方案：针孔近似

在 Carla 中直接使用 FOV=120° 的针孔相机渲染。等效焦距：

```
f = (W/2) / tan(HFOV/2) = 960 / tan(60°) ≈ 554.3 px
```

**差异分析**：
- 针孔模型：直线投影，边缘无畸变，但 120° 时边缘物体有明显透视拉伸
- 鱼眼模型：等距投影或等面积投影，边缘有桶形畸变，但保留更多信息

**可接受性**：
- 车道线检测主要关注图像中下部区域（h_samples 160~710，即 y=22%~99%），该区域透视畸变较小
- 训练和推理使用一致的相机模型即可，不需要与真实鱼眼完全一致
- 后续可通过 OpenCV 添加合成鱼眼畸变作为数据增强

### 后续增强：合成鱼眼畸变

如需更接近真实鱼眼效果，可在后处理阶段添加：

```python
# 使用 OpenCV 鱼眼模型
K_fisheye = K_pinhole.copy()
D = np.array([k1, k2, k3, k4])  # 鱼眼畸变系数
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_fisheye, D, np.eye(3), K_fisheye, (W, H), cv2.CV_16SC2)
distorted_img = cv2.remap(pinhole_img, map1, map2, cv2.INTER_LINEAR)
```

这一步骤是可选的数据增强，不影响核心标注流程。

### FOV 与远场分辨率：ROI 裁剪

#### 问题

120° HFOV 在 640×360 网络输入分辨率下，远场车道线信息严重不足：

| 距离 | 车道宽(3.5m) 原图 1920px | 直接 resize 到 640×360 | 车道标线(0.15m) 640×360 |
|------|------------------------|-----------------------|------------------------|
| 50m  | 38.8 px                | 13.0 px               | 0.56 px                |
| 100m | 19.4 px                | **6.5 px**            | 0.28 px (不可见)        |
| 150m | 12.9 px                | **4.3 px**            | 0.19 px (不可见)        |

核心矛盾：120° FOV 把像素预算分散到了极宽的视角，远场车道线信息被稀释到几乎不可用。

#### 方案：可配置 ROI 裁剪

Phase 1 保持 120° FOV 渲染（保留原始数据最大信息量），Phase 4 从 1920×1080 中心裁剪出较窄有效 FOV 的 ROI 区域，保持 16:9 宽高比，缩放到 1280×720。部署时对物理 120° 相机做同样的裁剪。

```
原始 1920×1080 (HFOV=120°)
┌──────────────────────────────────────────┐
│            ┌──────────────┐              │
│  丢弃 25°  │  ROI 裁剪     │  丢弃 25°   │
│            │  HFOV ≈ 70°  │              │
│            │  776×436      │              │
│            └──────────────┘              │
└──────────────────────────────────────────┘
                    ↓ resize
              1280×720 (TuSimple)
                    ↓ resize (训练时)
               640×360 (网络输入)
```

不同裁剪 FOV 的远场分辨率对比（100m 处车道宽 3.5m，最终 640px 宽）：

| 裁剪 HFOV | 裁剪宽度(px) | 100m 车道宽 | 相对增益 |
|-----------|-------------|------------|---------|
| 120° (不裁) | 1920       | 6.5 px     | 1.0×    |
| 80°       | 930         | 13.4 px    | 2.1×    |
| **70°**   | **776**     | **16.0 px** | **2.5×** |
| 60°       | 640         | 19.4 px    | 3.0×    |

**推荐裁剪 HFOV ≈ 70°**：远场分辨率提升 2.5 倍，同时保留约 70° 视野，足以覆盖 4 车道宽度。

裁剪区域的垂直位置使用**固定的名义 pitch (4°)** 计算，训练和部署共用同一裁剪参数。由于训练数据覆盖 0°~7° 的 pitch 范围，地平线在裁剪图中自然分布在 ~24%~41% 之间，模型学会适应不同地平线位置，无需部署时精确标定 pitch。

```
NOMINAL_PITCH = 4°, 固定 crop_y = 370

pitch =  0° → 地平线在裁剪图 39.0% 处, 道路占 61.0%
pitch =  4° → 地平线在裁剪图 30.1% 处, 道路占 69.9% ← 名义
pitch =  7° → 地平线在裁剪图 23.4% 处, 道路占 76.6%
```

详见 `config.py:compute_crop_params()`。

## 二、文件结构

```
tools/dashcam/tusimple/
├── __init__.py
├── config.py              # 相机参数、高度定义、TuSimple 常量
├── carla_world.py         # TuSimpleCarlaWorld（H0 参考双目 + H1~H6 Mono）
├── collect.py             # Phase 1: 数据采集（单 session）
├── run_full_collection.py # Phase 1: 批量采集（场景×姿态矩阵，自动重试/续采）
├── annotate_3d.py         # Phase 2: 3D 标注（modeld 推理 + 高度变换）
├── clean_and_sample.py    # Phase 3: 数据清洗与抽样（质量过滤 + 时间抽样 + 划分）
├── project_tusimple.py    # Phase 4: 3D→2D 投影 + TuSimple 格式输出
├── projection.py          # 3D→2D 投影核心算法库
├── viz_collect.py         # Phase 1 可视化: 多相机网格视图
├── viz_annotate.py        # Phase 2 可视化: 3D 标注投影叠加
├── viz_clean.py           # Phase 3 可视化: 清洗统计 + 拒绝帧示例
├── viz_tusimple.py        # Phase 4 可视化: TuSimple 2D 标注 + ROI 裁剪对比
```

## 三、数据目录结构

### 3.1 Phase 1 输出 — 原始采集数据

```
data/tusimple/<session_tag>/
├── clip_info.json           # 会话元数据（相机参数、高度、场景信息）
├── H0/                      # openpilot 参考双目相机（1.22m）
│   ├── road_000000.png      # 1928×1208 窄角 (FOV=40°)
│   ├── wide_000000.png      # 1928×1208 广角 (FOV=120°)
│   └── metadata.jsonl       # 每帧元数据（v_ego, world_pose, camera_height）
├── H1/                      # Mono 单目相机（1.22m）
│   ├── 000000.png           # 1920×1080 (FOV=120°)
│   └── metadata.jsonl
├── H2/                      # Mono 单目相机（1.30m）
│   ├── 000000.png
│   └── metadata.jsonl
├── H3/ ... H6/              # 同上
```

### 3.2 Phase 2 输出 — 3D 标注

```
data/tusimple/<session_tag>/3d_labels/
├── clip_info.json           # 标注元数据（含 source_session_dir）
├── H1/                      # H1 高度的 3D 车道线标注
│   ├── 000000.json
│   └── ...
├── H2/ ... H6/              # 其他高度（经高度变换）
```

### 3.3 Phase 3 输出 — 清洗与抽样结果

```
data/tusimple/<session_tag>/splits/
├── clean_log.txt            # 通过质量过滤的帧列表
├── sampled_log.txt          # 时间抽样后的帧列表
├── train.txt                # 训练集帧列表
├── val.txt                  # 验证集帧列表
├── test.txt                 # 测试集帧列表
└── stats.json               # 清洗统计（总帧数、通过率、各高度分布）
```

帧列表格式 (每行一条): `<height_tag>/<frame_id>`，例如 `H1/000000`

### 3.4 Phase 4 输出 — TuSimple 格式

```
data/tusimple/<session_tag>/tusimple/
├── H1/
│   ├── images/              # 1280×720 JPEG（从 Mono 图像 ROI 裁剪 + 缩放）
│   │   ├── 000000.jpg
│   │   └── ...
│   └── labels.json          # TuSimple JSON Lines
├── H2/ ... H6/              # 同上结构
├── train.json               # 训练集 TuSimple 标签（合并所有高度）
├── val.json                 # 验证集 TuSimple 标签
└── test.json                # 测试集 TuSimple 标签
```

### 3.5 clip_info.json 示例

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
  "h0_camera": {
    "tag": "H0",
    "height": 1.22,
    "narrow": {"width": 1928, "height": 1208, "fov": 40, "focal": 2648.0},
    "wide":   {"width": 1928, "height": 1208, "fov": 120, "focal": 567.0}
  },
  "mono_camera": {
    "width": 1920, "height": 1080, "fov": 120,
    "focal": 554.3,
    "model": "pinhole",
    "format": "jpeg"
  },
  "tusimple_output": {
    "width": 1280, "height": 720,
    "h_samples": [160, 170, "...", 710],
    "crop_hfov": 70,
    "nominal_pitch": 4.0,
    "crop_rect": [572, 370, 776, 436],
    "effective_focal": 914.3
  },
  "heights": {
    "H1": 1.22, "H2": 1.30, "H3": 1.50,
    "H4": 2.00, "H5": 2.50, "H6": 3.00
  },
  "save_every": 4,
  "simulation": {
    "fps": 20.0,
    "fixed_delta_seconds": 0.05,
    "num_npc": 40
  }
}
```

## 四、各文件详细设计

### 4.1 `config.py` — 配置中心

定义所有相机参数、高度常量和 TuSimple 格式常量。

```python
import numpy as np
from dataclasses import dataclass

# ── H0: openpilot 参考双目相机 (标注基准) ──
H0_HEIGHT = 1.22             # 安装高度 (meters)
H0_W, H0_H = 1928, 1208
H0_NARROW_FOV = 40           # degrees
H0_WIDE_FOV = 120            # degrees
H0_NARROW_FOCAL = 2648.0
H0_WIDE_FOCAL = 567.0

# ── H1~H6: Mono 单目相机 (Carla 针孔渲染) ──
MONO_W, MONO_H = 1920, 1080
MONO_HFOV = 120  # degrees
MONO_FOCAL = MONO_W / 2 / np.tan(np.radians(MONO_HFOV / 2))  # ≈554.256

K_MONO = np.array([
  [MONO_FOCAL, 0.0,       MONO_W / 2],
  [0.0,        MONO_FOCAL, MONO_H / 2],
  [0.0,        0.0,        1.0],
])

# ── TuSimple 输出 ──
TUSIMPLE_W, TUSIMPLE_H = 1280, 720

# TuSimple 标准 h_samples: v=160 到 v=710, 步长 10
TUSIMPLE_H_SAMPLES = list(range(160, 720, 10))  # 56 个采样点

# ── ROI 裁剪参数 (120° → 目标 FOV，提升远场分辨率) ──
CROP_HFOV = 70        # degrees, 裁剪后有效水平视场角
NOMINAL_PITCH = 4.0   # degrees, 名义安装 pitch (用于固定裁剪区域)
HORIZON_RATIO = 0.3   # 地平线在裁剪图中的垂直位置 (0=顶部, 1=底部)

def compute_crop_params(
  pitch_deg: float = NOMINAL_PITCH,
  crop_hfov: float = CROP_HFOV,
  horizon_ratio: float = HORIZON_RATIO,
) -> dict:
  """计算 ROI 裁剪参数和等效内参。

  从 1920×1080 (HFOV=120°) 中裁剪出 crop_hfov 的中心区域，
  保持 16:9 宽高比，然后缩放到 1280×720。

  使用固定的 NOMINAL_PITCH 计算裁剪区域，训练和部署共用。
  不同 session 的实际 pitch 导致地平线在裁剪图中自然浮动，
  增强模型对 pitch 变化的鲁棒性。

  Returns:
    {
      'crop_rect': (x, y, w, h),    # 在 1920×1080 中的裁剪矩形
      'crop_hfov': float,            # 裁剪后有效 HFOV (degrees)
      'K_crop': np.ndarray,          # 裁剪+缩放后的 3×3 等效内参 (1280×720)
      'effective_focal': float,      # TuSimple 分辨率下的等效焦距
    }
  """
  # 裁剪尺寸 (像素)
  crop_w = int(2 * MONO_FOCAL * np.tan(np.radians(crop_hfov / 2)))
  crop_h = crop_w * TUSIMPLE_H // TUSIMPLE_W   # 保持 16:9

  # 水平居中
  crop_x = (MONO_W - crop_w) // 2

  # 垂直: 根据名义 pitch 确定地平线位置，再按 horizon_ratio 放置
  horizon_y = MONO_H / 2 - MONO_FOCAL * np.tan(np.radians(pitch_deg))
  crop_y = int(horizon_y - horizon_ratio * crop_h)
  crop_y = max(0, min(crop_y, MONO_H - crop_h))

  # 等效内参 (裁剪区域 → 1280×720)
  scale = TUSIMPLE_W / crop_w
  f_crop = MONO_FOCAL * scale
  cx_crop = (MONO_W / 2 - crop_x) * scale    # 水平居中裁剪时 = TUSIMPLE_W / 2
  cy_crop = (MONO_H / 2 - crop_y) * scale

  K_crop = np.array([
    [f_crop, 0.0,    cx_crop],
    [0.0,    f_crop, cy_crop],
    [0.0,    0.0,    1.0],
  ])

  return {
    'crop_rect': (crop_x, crop_y, crop_w, crop_h),
    'crop_hfov': crop_hfov,
    'K_crop': K_crop,
    'effective_focal': f_crop,
  }

# 固定裁剪参数 (NOMINAL_PITCH=4°, CROP_HFOV=70°):
#   crop_rect = (572, 370, 776, 436)
#   K_crop = [[914.3, 0, 640.0], [0, 914.3, 280.4], [0, 0, 1]]
#   effective_focal = 914.3

# ── 高度定义 ──
HEIGHT_DEFS = {
  'H1': 1.22,   # 标准轿车 (与 H0 同高度)
  'H2': 1.30,
  'H3': 1.50,   # 中型 SUV
  'H4': 2.00,
  'H5': 2.50,
  'H6': 3.00,   # 卡车驾驶室
}

@dataclass
class CameraSlotConfig:
  tag: str       # 'H1', ..., 'H6'
  height: float  # meters above ground
```

### 4.2 `carla_world.py` — TuSimpleCarlaWorld

**参考模板**: `tools/dashcam/carla_multi_height_world.py`

独立的 Carla 世界管理类，同时管理 H0 参考双目和 H1~H6 Mono 单目相机。

**相机布局**:

| 编号 | 类型 | 分辨率 | FOV | 高度 |
|------|------|--------|-----|------|
| H0 narrow | 参考窄角 | 1928×1208 | 40° | 1.22m |
| H0 wide | 参考广角 | 1928×1208 | 120° | 1.22m |
| H1 mono | 单目 | 1920×1080 | 120° | 1.22m |
| H2 mono | 单目 | 1920×1080 | 120° | 1.30m |
| ... | ... | ... | ... | ... |
| H6 mono | 单目 | 1920×1080 | 120° | 3.00m |

总计 2 + 6 = 8 个相机传感器，全部共享相同的 pitch/yaw 安装角。

```python
class TuSimpleCarlaWorld:
  """Carla world with H0 reference cameras + H1~H6 mono cameras.

  H0 (openpilot narrow+wide at 1.22m) provides modeld 3D annotation.
  H1~H6 (1920×1080, FOV=120°) at various heights provide TuSimple training images.
  """

  def __init__(
    self,
    host: str = '127.0.0.1',
    port: int = 2000,
    town: str = 'Town04',
    weather: str = 'ClearNoon',
    spawn_point: int = 16,
    random_spawn: bool = False,
    camera_pitch_deg: float = 5.0,
    camera_yaw_deg: float = 0.0,
    camera_forward_offset_m: float = 0.8,
    mono_heights: list[CameraSlotConfig] | None = None,
    num_npc: int = 40,
    high_quality: bool = False,
    speed_range: tuple[float, float] = (40.0, 100.0),
    speed_interval: tuple[float, float] = (8.0, 20.0),
  ):
    """Initialize Carla world with H0 reference + H1~H6 mono cameras.

    Camera creation:
      1. H0 road: FOV=40, 1928×1208, height=1.22m
      2. H0 wide: FOV=120, 1928×1208, height=1.22m
      3. H1~H6 mono: FOV=120, 1920×1080, height per slot

    All cameras share the same pitch/yaw rotation:
      carla.Rotation(pitch=-camera_pitch_deg, yaw=-camera_yaw_deg)
    (openpilot→Carla 坐标系转换: 取反)
    """

  def get_frames(self) -> dict | None:
    """Return synchronized frames when ALL cameras have same frame_id.

    Returns:
      {
        'h0_road': np.ndarray,  # (1208, 1928, 3) RGB
        'h0_wide': np.ndarray,  # (1208, 1928, 3) RGB
        'mono': {
          'H1': np.ndarray,     # (1080, 1920, 3) RGB
          'H2': np.ndarray,
          ...
        }
      }
      or None if not yet synchronized.
    """

  def tick(self) -> None: ...
  def get_vehicle_speed(self) -> float: ...
  def get_vehicle_transform(self): ...
  def get_clip_metadata(self, session_id: str) -> dict: ...
  def close(self) -> None: ...
```

**帧同步实现**: 内部维护 `_latest` 字典，每个相机回调写入 `(frame_id, rgb_array)`。`get_frames()` 检查所有 8 个相机的 frame_id 是否一致。Carla 同步模式 + sensor_tick 保证帧对齐。

### 4.3 `collect.py` — Phase 1: 数据采集

**职责**: 仅负责从 Carla 采集原始图像和元数据，不做任何标注。

**工作流**:

1. 解析 CLI 参数
2. **磁盘空间检查**: 剩余空间 < `MIN_DISK_FREE_GB` (默认 20 GB) → 拒绝启动并报错
3. 创建 `TuSimpleCarlaWorld` (连接 Carla, 生成自车+NPC, 挂载 H0+H1~H6 相机)
4. 保存 `clip_info.json`
5. 主循环:
   - `world.tick()` → `world.get_frames()` 等待帧同步
   - `save_every` 跳帧 (默认 4, 即有效 5 FPS)
   - 保存 H0 参考帧: `H0/road_XXXXXX.png`, `H0/wide_XXXXXX.png`
   - 保存 H1~H6 Mono 帧: `H*/XXXXXX.png`
   - 写各目录的 metadata.jsonl (v_ego, world_pose, camera_height)
   - **周期性磁盘检查** (每 500 帧): 剩余空间 < 20 GB → 提前停止并打印警告
6. 清理: 等待异步写完成, 关闭 Carla

**CLI 接口**:

```bash
python tools/dashcam/tusimple/collect.py \
  --heights H1 H2 H3 H4 H5 H6 \
  --map Town04 --weather ClearNoon \
  --pitch 5.0 --yaw 0.0 \
  --max-frames 500 --save-every 4 \
  --output-base data/tusimple \
  --num-npc 40 --spawn-point 16 --random-spawn \
  --speed-range 40 100 --speed-interval 8 20 \
  --no-display \
  --mono-jpeg-quality 95  # 可选: Mono 帧用 JPEG 节省磁盘
```

**输出**: `data/tusimple/<session_tag>/` 目录，包含 H0/ + H1~H6/ + clip_info.json

**`max_frames` 语义**: `--max-frames 500` 表示实际保存 500 帧到磁盘（与原有 `collect_multi_height.py` 一致）。配合 `--save-every 4`，每 4 个 tick 保存一帧，共需 ~2000 个 tick，帧 ID 为 000000, 000004, 000008, ...

**磁盘估算** (单 session, 500 帧, H1~H6):
- H0 PNG: ~5 MB/帧 × 2 = 10 MB/帧 → ~5 GB
- Mono PNG: ~2 MB/帧 × 6 = 12 MB/帧 → ~6 GB
- Mono JPEG (Q=95): ~0.5 MB/帧 × 6 = 3 MB/帧 → ~1.5 GB
- 单 session 合计: ~6.5 GB (JPEG) 或 ~11 GB (PNG)
- 推荐使用 `--mono-jpeg-quality 95` 节省磁盘
- **磁盘保护**: 剩余空间 < 20 GB 时自动停止

#### `run_full_collection.py` — 批量数据采集

**参考模板**: `tools/dashcam/run_full_collection.py`

对 `collect.py` 的批量封装。遍历所有场景（地图×天气）和相机姿态（pitch×yaw）组合，为每个组合自动运行一次完整采集。

**场景矩阵**:

```python
SCENES = [
  ('Town04', 'ClearNoon'),
  ('Town04', 'ClearSunset'),
  ('Town04', 'CloudyNoon'),
  ('Town04', 'WetNoon'),
  # 后续可扩展更多地图和天气
  # ('Town03', 'ClearNoon'),
  # ('Town05', 'CloudySunset'),
]
```

**相机姿态网格** (pitch × yaw):

```
pitch \ yaw  | -3°  | -1.5° |  0°  | +1.5° | +3°  | 采样点
-------------|------|-------|------|-------|------|------
 0°   (水平) |  ○   |       |  ○   |       |  ○   |  3
 1°          |      |  ○    |  ○   |  ○    |      |  3
 2°          |  ○   |  ○    |  ○   |  ○    |  ○   |  5
 3°          |  ○   |  ○    |  ○   |  ○    |  ○   |  5
 4° (名义)   |  ○   |  ○    |  ●   |  ○    |  ○   |  5+1
 5°          |  ○   |  ○    |  ○   |  ○    |  ○   |  5
 6°          |  ○   |  ○    |  ○   |  ○    |  ○   |  5
 7°          |      |  ○    |  ○   |  ○    |      |  3
                                              合计: 34 姿态
```

- ○ = 普通采样点 (1× max_frames)
- ● = 名义中心 (4°, 0°), 采集 2× max_frames
- 范围 0°~7°: 覆盖轿车 (3°~7°) 和卡车 (0°~4°) 安装场景
- 总计: 4 场景 × 34 姿态 = **136 个 session**

**关键特性** (复用 `run_full_collection.py` 的成熟机制):

| 特性 | 说明 |
|------|------|
| 断点续采 | 基于帧文件数自动跳过已完成 session，部分采集从断点恢复 |
| 自动重试 | Carla 崩溃后等待 N 秒重试，最多 M 次 (默认 30s × 3) |
| 磁盘检查 | 每个 session 开始前检查剩余空间，低于阈值自动停止 |
| 优雅中断 | Ctrl+C 完成当前 session 后停止，再次 Ctrl+C 强制退出 |
| 进度日志 | `collection_progress.json` 记录每个 session 的状态 |
| 随机出生点 | `--random-spawn` 确保每次运行的起始位置不同 |

**CLI 接口**:

```bash
# 列出所有 session 及其完成状态（不采集）
python tools/dashcam/tusimple/run_full_collection.py --list

# 启动（或恢复）批量采集
python tools/dashcam/tusimple/run_full_collection.py \
  --output-base data/tusimple \
  --max-frames 500 --save-every 4 \
  --num-npc 40 --no-display \
  --mono-jpeg-quality 95

# 崩溃后恢复 — 重新运行同一命令即可
python tools/dashcam/tusimple/run_full_collection.py \
  --output-base data/tusimple \
  --max-frames 500 --save-every 4 \
  --num-npc 40 --no-display

# 跳过前 50 个 session（手动指定起点）
python tools/dashcam/tusimple/run_full_collection.py \
  --output-base data/tusimple --start-from 50
```

**session 命名**: `{map}_{weather}_p{pitch}_y{yaw}`，例如 `Town04_ClearNoon_p5.0_y0.0`

**磁盘估算** (136 session × 500 保存帧/session):
- 使用 `--mono-jpeg-quality 95`: 500 帧 × (10 + 3) MB/帧 ≈ 6.5 GB/session
- 全量 (6 高度): 136 × 6.5 GB ≈ **884 GB** (在 1.3 TB 可用空间内)
- 精简 (3 高度 H1/H3/H6): 136 × 4 GB ≈ **544 GB**
- **磁盘保护**: 每个 session 开始前检查，剩余 < 20 GB 自动停止

### 4.4 `annotate_3d.py` — Phase 2: 3D 标注

**职责**: 读取 H0 参考相机图像，运行 openpilot modeld 推理，生成 3D 车道线标注。对 H2~H6 执行高度变换。

**输入**: Phase 1 输出的 `data/tusimple/<session_tag>/` 目录
**输出**: `data/tusimple/<session_tag>/3d_labels/` 目录

**幂等性**: 已存在的 3D 标注 JSON 文件自动跳过，支持断点恢复。

**复用代码**: 从 `annotate_multi_height.py` 直接导入以下函数:
- `decode_model_output()` — 解码模型输出
- `transform_annotation()` — 高度变换
- `_load_or_compile_model()` — 加载/编译 tinygrad 模型
- `_run_inference()` — 运行推理

**工作流**:

```
1. 读取 clip_info.json → pitch_deg, yaw_deg → rpyCalib = [0.0, pitch_rad, yaw_rad]
2. 计算 warp 矩阵 (使用 H0 openpilot 标准内参):
   warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
   warp_wide = compute_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True)
3. 加载 tinygrad 模型 (auto-compile ONNX)
4. 遍历帧:
   a. 从 H0/ 读取 road + wide PNG
   b. 预处理 (GPU OpenCL or CPU) → (6, 128, 256) uint8 YUV
   c. 拼接时序帧 [prev, curr] → (12, 128, 256)
   d. 推理 → decode_model_output() → canonical 3D annotation (H0 高度)
   e. 质量过滤: lane_lines_prob[1] > min_prob AND lane_lines_prob[2] > min_prob
   f. 对每个目标高度 H1~H6:
      - H1: H0 和 H1 同高度 (1.22m)，直接使用 canonical
      - H2~H6: transform_annotation(canonical, h1=1.22, h_k=h_k)
   g. 保存 3d_labels/H*/XXXXXX.json
```

**3D 标注 JSON 格式** (与 annotate_multi_height 兼容):

```json
{
  "frame_id": "000000",
  "camera_height": 1.22,
  "v_ego": 15.3,
  "world_pose": [100.2, -5.1, 0.3, 0.0, 0.0, 45.2],
  "label_source": "pretrained_h0",
  "ll_quality_pass": true,
  "lane_lines": [[[0.0, -3.5, 1.22], [0.19, -3.5, 1.22], ...], ...],
  "lane_lines_prob": [0.95, 0.98, 0.97, 0.42],
  "road_edges": [[[0.0, -5.0, 1.22], ...], ...],
  "lead": [...],
  "pose": [15.3, 0.0, 0.0, 0.0, 0.0, 0.01],
  "road_transform": [0.0, 0.0, 1.22, 0.0, 0.0, 0.0],
  "wide_from_device_euler": [0.0, 0.0, 0.0]
}
```

**CLI 接口**:

```bash
python tools/dashcam/tusimple/annotate_3d.py \
  data/tusimple/Town04_ClearNoon_p5.0_y0.0/ \
  --onnx selfdrive/modeld/models/driving_vision.onnx \
  --heights H1 H3 H6 \
  --min-ll-prob 0.3 \
  --output data/tusimple/Town04_ClearNoon_p5.0_y0.0/3d_labels/ \
  --no-gpu-preprocess  # 可选: 禁用 GPU 预处理
```

### 4.5 `projection.py` — 3D→2D 投影核心算法库

这是系统的数学核心，供 `project_tusimple.py` 调用。

#### 4.5.1 坐标系和投影公式

```
openpilot 标定坐标系 (calibration frame):
  x = forward (前方)
  y = right   (右侧)
  z = down    (向下)

投影管线:
  3D 标定点 → device_from_calib 旋转 → view_from_device 变换 → 内参投影 → 2D 像素

  pixel = K_mono @ view_frame_from_device_frame @ rot_from_euler(rpyCalib) @ point_3d

其中:
  view_frame_from_device_frame = [[0,0,1],[1,0,0],[0,1,0]]  # x_view=z_dev, y_view=x_dev, z_view=y_dev
  rpyCalib = [0.0, pitch_rad, yaw_rad]                       # 相机安装角
```

#### 4.5.2 核心函数

```python
def project_3d_to_mono(
  points_3d: np.ndarray,    # (N, 3) [x, y, z] in calibration frame
  K_mono: np.ndarray,       # 3×3 Mono 相机内参 (渲染分辨率)
  rpyCalib: np.ndarray,     # [roll, pitch, yaw] 弧度
) -> np.ndarray:
  """将标定坐标系中的 3D 点投影到 Mono 相机像素坐标。

  投影公式与 visualizer.py:project_points_to_image() 完全一致:
    T = K_mono @ view_frame_from_device_frame @ rot_from_euler(rpyCalib)
    proj = T @ [x, y, z]^T
    pixel = proj[:2] / proj[2]

  相机后方的点 (proj[2] <= 0) 返回 NaN。

  Args:
    points_3d: 3D 点坐标, shape (N, 3)
    K_mono: 内参矩阵
    rpyCalib: 标定欧拉角 [roll, pitch, yaw]

  Returns:
    (N, 2) float array, 每行 [u, v] 像素坐标, 无效点为 NaN
  """
```

```python
def resample_lane_at_h_samples(
  uv: np.ndarray,             # (M, 2) 投影后的像素点 (已缩放到输出分辨率)
  h_samples: list[int],       # TuSimple v 坐标采样列表
  x_range: tuple[int, int] = (0, 1279),
) -> list[int]:
  """在 TuSimple h_samples 的每个 v 值处插值 x 坐标。

  前置条件: 输入 v 已单调（由 lanes_3d_to_tusimple 在调用前做裁断）。

  算法:
    1. 滤除 NaN 和无效点
    2. 去除重复 v 值 (保留第一个)
    3. 对每个 h_sample:
       - 若 v 超出投影多段线的 v 范围 → -2
       - np.interp() 线性插值 u → 取整
       - 若 x 超出 x_range → -2
       - 否则 → int(round(x))

  Returns:
    长度为 len(h_samples) 的整数列表, -2 表示不可见
  """
```

```python
def lanes_3d_to_tusimple(
  lane_lines_3d: np.ndarray,    # (4, 33, 3) 4 条车道线
  lane_lines_prob: np.ndarray,  # (4,) 车道线存在概率
  road_edges_3d: np.ndarray,    # (2, 33, 3) 2 条道路边缘
  K_tusimple: np.ndarray,       # 3×3 等效内参 (TuSimple 输出分辨率 1280×720)
  rpyCalib: np.ndarray,         # [roll, pitch, yaw]
  h_samples: list[int],         # TuSimple h_samples
  min_prob: float = 0.3,        # 最小概率阈值
  min_visible_pts: int = 2,     # 最少可见采样点
  min_x_distance: float = 2.0,  # 最小前方距离 (过近的点不稳定)
) -> list[list[int]]:
  """完整的 3D 车道线到 TuSimple 2D 格式转换。

  K_tusimple 是 ROI 裁剪 + 缩放后的等效内参 (由 compute_crop_params() 生成)，
  3D 点直接投影到 1280×720 TuSimple 坐标系，无需额外缩放。

  处理流程 (per lane):
    1. 检查概率 > min_prob
    2. 过滤 x < min_x_distance 的近场点
    3. project_3d_to_mono(pts, K_tusimple) 直接投影到 TuSimple 分辨率
    4. 单调性裁断: 沿 3D 距离 (近→远) 检查 v 单调递减，折返处截断
    5. resample_lane_at_h_samples() 插值
    6. 过滤可见点数 < min_visible_pts 的短车道线

  Returns:
    list of lanes, 每条 lane 是 len(h_samples) 的整数列表
    空列表表示无有效车道线
  """
```

#### 4.5.3 TuSimple 插值算法详解

```python
# 以一条车道线为例:
lane_3d = lane_lines_3d[i]  # (33, 3) = [x, y, z]

# Step 1: 过滤近场
mask = lane_3d[:, 0] >= min_x_distance  # x > 2m
pts = lane_3d[mask]  # (M, 3), M <= 33, 已按 x 从近到远排列

# Step 2: 用 K_crop 直接投影到 TuSimple 分辨率 (1280×720)
# K_crop 已包含 ROI 裁剪 + 缩放的等效内参，无需额外 scale
uv_tusimple = project_3d_to_mono(pts, K_crop, rpyCalib)  # (M, 2)

# Step 3: 滤除无效点 (NaN, 相机后方)
valid = ~np.isnan(uv_tusimple[:, 0])
u = uv_tusimple[valid, 0]
v = uv_tusimple[valid, 1]

# Step 4: 单调性裁断 (弯道处理)
# 3D 点按 x 从近到远排列，投影后 v 应从大 (图像底) 到小 (图像顶) 单调递减
# 若在某处 v 开始回升（弯道折返），截断该点之后的远端部分
#
# 示例:
#   正常直线:  v = [680, 600, 520, 440, 360, 280]  ← 单调递减 ✓
#   弯道折返:  v = [680, 600, 520, 480, 510, 530]  ← 在 index=3 处开始回升
#                                    ↑ 截断点: 保留 [680, 600, 520, 480]
#
if len(v) > 1:
  cutoff = len(v)
  for j in range(1, len(v)):
    if v[j] >= v[j - 1]:  # v 不再递减 → 折返点
      cutoff = j
      break
  u = u[:cutoff]
  v = v[:cutoff]

# Step 5: 按 v 排序 (升序，供 np.interp 使用)
idx = np.argsort(v)
v_sorted, u_sorted = v[idx], u[idx]

# Step 6: 去除重复 v 值
unique_mask = np.diff(v_sorted, prepend=-1) > 0
v_sorted = v_sorted[unique_mask]
u_sorted = u_sorted[unique_mask]

# Step 7: 在每个 h_sample 处插值
lane_x = []
for h in h_samples:
  if len(v_sorted) < 2 or h < v_sorted[0] or h > v_sorted[-1]:
    lane_x.append(-2)
  else:
    x_interp = np.interp(h, v_sorted, u_sorted)
    x_int = int(round(x_interp))
    if 0 <= x_int < TUSIMPLE_W:
      lane_x.append(x_int)
    else:
      lane_x.append(-2)
```

**边界情况处理**:

- **弯道折返** (核心处理):

```
直线道路投影:                     弯道投影 (折返):
  远 ·                              远 · ← 被截断
     ·                                ·  ↙ 折返
     ·                              · ← 截断点
     ·                             ·
     ·                            ·
  近 · (图像底部)              近 · (图像底部)

  v 单调递减 ✓                   v 在折返处停止
  → 完整车道线                    → 车道线到折返点为止
```

  投影后的 2D 车道线沿 3D 距离（近→远）v 应单调递减。大曲率弯道可能导致远端点投影回到图像下方（v 回升），此时在折返点截断，保证：
  1. `np.interp` 的输入 v 严格单调，插值数学正确
  2. 车道线不会在图像上出现"回头"形状
  3. 截断后的车道线仍然连续、可用，只是比直线时短

- **近场盲区**: 高度越大，近场盲区越大（路面在相机正下方）。通过 `min_x_distance` 过滤
- **远场消失**: 超出 192m 或投影点过于密集，由 h_samples 范围自然截断

### 4.6 `clean_and_sample.py` — Phase 3: 数据清洗与抽样

**职责**: 对 Phase 2 输出的 3D 标注进行质量过滤、时间抽样和训练/验证/测试集划分。

**输入**: Phase 2 输出的 `data/tusimple/<session>/3d_labels/` 目录
**输出**: `data/tusimple/<session>/splits/` 目录

**幂等性**: Phase 3 输出为整体文件 (train.txt 等)，重新运行时覆盖写入（非增量），天然幂等。

**参考**: `tools/dashcam/train/clean_data.py` 和 `tools/dashcam/train/split_dataset.py`

**三步流程**:

#### Step 1: 质量过滤

质量过滤依赖H0的数据，H1～H6过滤结果相同。

```
对每帧 3D 标注 JSON:
  - 过滤每个session的第0帧，该帧标注质量不高
  - 检查 ll_quality_pass 标志
  - 检查内侧车道线概率: lane_lines_prob[1] > prob_threshold AND lane_lines_prob[2] > prob_threshold
  - 检查自车速度: v_ego > min_speed (过滤停车帧)
  - 通过 → 写入 clean_log.txt的内容“<frame id>, pass”
  - 不通过 → 写入 clean_log.txt的内容“<frame id>, fail”
```

**过滤条件** (可配置):
- `--min-ll-prob 0.7`: 内侧车道线最低概率
- `--min-speed 1.0`: 最低自车速度 (m/s)，过滤停车/起步帧
- `--require-both-inner`: 要求左右内侧车道线同时存在 (默认 true)

#### Step 2: 时间抽样

```
对 clean_log.txt 中的帧:
  - 每 N 帧取 1 帧 (默认 N=5, 即 1 FPS 等效)
  - 确保相邻帧有足够时间间隔，避免训练数据时序冗余
  - 输出 → sampled_log.txt, "<frame id>"
```

**抽样参数**:
- `--sample-every 5`: 每 5 帧取 1 帧
- `--per-height`: 各高度独立抽样 (默认 true)

#### Step 3: 训练/验证/测试划分

```
对 sampled_log.txt 中的帧:
  - 全局随机打乱
  - 按比例划分: train:val:test = 8:1:1 (默认)
  - 写入 train.txt, val.txt, test.txt
  - 输出统计到 stats.json
```

**划分参数**:
- `--split-ratio 0.8 0.1 0.1`: 训练/验证/测试比例
- `--seed 42`: 随机种子 (可复现)

**CLI 接口**:

```bash
python tools/dashcam/tusimple/clean_and_sample.py \
  data/tusimple/Town04_ClearNoon_p5.0_y0.0/ \
  --heights H1 H3 H6 \
  --min-ll-prob 0.3 \
  --min-speed 1.0 \
  --sample-every 5 \
  --split-ratio 0.8 0.1 0.1 \
  --seed 42
```

**stats.json 示例**:

```json
{
  "total_frames": 1500,
  "quality_pass": 1260,
  "quality_pass_rate": 0.84,
  "sampled": 252,
  "per_height": {
    "H1": {"pass": 420, "sampled": 84, "train": 67, "val": 9, "test": 8},
    "H3": {"pass": 420, "sampled": 84, "train": 67, "val": 9, "test": 8},
    "H6": {"pass": 420, "sampled": 84, "train": 68, "val": 8, "test": 8}
  },
  "train": 202,
  "val": 26,
  "test": 24
}
```

### 4.7 `project_tusimple.py` — Phase 4: 2D 投影

**职责**: 读取 Phase 3 的sample_log帧列表、Phase 2 的 3D 标注和 Phase 1 的 Mono 图像，执行 ROI 裁剪 + 3D→2D 投影，输出 TuSimple 2D 格式。Phase 4 **不做质量过滤**（已由 Phase 3 完成），仅处理投影层面的安全检查。

**输入**:
- Phase 1 输出: `data/tusimple/<session>/H1~H6/` (Mono 图像, 1920×1080)
- Phase 2 输出: `data/tusimple/<session>/3d_labels/H1~H6/` (3D 标注)
- Phase 3 输出: `data/tusimple/<session>/splits/` (帧列表)

**输出**: `data/tusimple/<session>/tusimple/` (TuSimple JSON + JPEG)

**幂等性**: 已存在的输出文件自动跳过，支持断点恢复。

**车道线选择策略** (TuSimple 固定 4 条):

```
TuSimple 槽位:      L-out  | L-inn | R-inn | R-out
openpilot lane_line: [0]    | [1]   | [2]   | [3]
openpilot road_edge: [0]    |  —    |  —    | [1]
选择逻辑:           LL优先  | LL    | LL    | LL优先
                    低置信   |       |       | 低置信
                    →RE补位  |       |       | →RE补位
```

对于每个槽位:
1. 若 lane_line prob > `lane_prob_threshold`(默认值0.2) → 使用 lane_line
2. 若 lane_line prob ≤ threshold 且该槽位有对应 road_edge (仅 L-out ← RE[0], R-out ← RE[1]) → 使用 road_edge
3. 否则 → 该槽位所有 h_sample 填 -2 (不可见)

注: L-inn (lane[1]) 和 R-inn (lane[2]) 无 road_edge 补位源。

**工作流**:

```
1. 读取 clip_info.json → rpyCalib, mono_format
2. 计算固定 ROI 裁剪参数 (使用 NOMINAL_PITCH=4°，与部署一致):
   crop_params = compute_crop_params()  # 使用默认 NOMINAL_PITCH
   → crop_rect, K_crop
3. 读取 splits/ 目录下的帧列表 (train.txt, val.txt, test.txt)
4. 对每个帧列表 (train/val/test):
   对每个帧 <height_tag>/<frame_id>:
     a. 检查输出文件已存在 → 跳过 (幂等性)
     b. 读取 3d_labels/<height_tag>/<frame_id>.json
     c. 按车道线选择策略选出 4 条车道线 (lane_line 优先, road_edge 补位)
     d. 调用 lanes_3d_to_tusimple(K_tusimple=K_crop):
        - 用 K_crop 直接投影到 1280×720 TuSimple 坐标系
        - 在 h_samples 处插值
     e. 若投影后所有 4 条车道线全部无效 → 跳过并记录 warning
     f. 读取 Mono 图像 (格式从 clip_info.json 获取) → ROI 裁剪 → resize 1280×720 → 保存 JPEG
     g. 追加 TuSimple JSON Line 到对应的 labels.json
5. 生成合并的 train.json, val.json, test.json (跨高度)
6. 输出统计 (总帧数, 有效帧数, 平均车道线数, 跳过帧数)
```

**图像预处理流程**:

```python
# Mono 图像 (1920×1080, HFOV=120°) → TuSimple 图像 (1280×720, HFOV≈70°)
x, y, w, h = crop_params['crop_rect']   # (572, 370, 776, 436)
roi = mono_img[y:y+h, x:x+w]           # 裁剪 ROI
tusimple_img = cv2.resize(roi, (TUSIMPLE_W, TUSIMPLE_H))  # 缩放到 1280×720
```

**CLI 接口**:

```bash
python tools/dashcam/tusimple/project_tusimple.py \
  data/tusimple/Town04_ClearNoon_p5.0_y0.0/ \
  --crop-hfov 70 \
  --lane-prob-threshold 0.2 \
  --min-visible-pts 2 \
  --output data/tusimple/Town04_ClearNoon_p5.0_y0.0/tusimple/
```

注意:
- 不再需要 `--heights` 参数，帧列表已包含高度信息
- 不做质量过滤（已在 Phase 3 完成），仅做投影层面安全检查
- 固定输出 4 条车道线 (L-out, L-inn, R-inn, R-out)，低置信度 lane_line 由 road_edge 补位
- `--lane-prob-threshold`: lane_line 使用/补位的置信度阈值 (默认 0.2)
- 已处理的帧自动跳过（幂等性），支持断点恢复

### 4.8 可视化工具

每个 Phase 有独立的可视化脚本，用于人工检查该阶段的输出质量。

**通用选项**: `--max-frames N` 限制显示帧数, `--output-dir` 保存图片, `--video` 导出视频

#### 4.8.1 `viz_collect.py` — Phase 1: 多相机采集检查

```bash
python tools/dashcam/tusimple/viz_collect.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --max-frames 20
```

**检查内容**:
- 8 相机网格视图: H0 road + H0 wide + H1~H6 mono 拼接在一张图中
- 每个子图叠加: 帧号、相机标签、安装高度
- 底部信息栏: v_ego (km/h)、world_pose

```
┌─────────────┬─────────────┐
│ H0 road     │ H0 wide     │
│ 1928×1208   │ 1928×1208   │
│ FOV=40°     │ FOV=120°    │
├──────┬──────┼──────┬──────┤
│ H1   │ H2   │ H3   │ H4   │
│1.22m │1.30m │1.50m │2.00m │
├──────┼──────┼──────┼──────┤
│ H5   │ H6   │ (info bar)  │
│2.50m │3.00m │ v=72km/h    │
└──────┴──────┴─────────────┘
```

**关键验证点**:
- 帧同步: 各相机画面内容一致（同一时刻）
- 高度差异: H1 vs H6 的视角变化符合物理预期
- NPC 车辆和道路场景是否正常

#### 4.8.2 `viz_annotate.py` — Phase 2: 3D 标注检查

```bash
python tools/dashcam/tusimple/viz_annotate.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --heights H1 H3 H6 \
  --max-frames 20
```

**检查内容**:
- 将 3D 车道线标注投影到对应高度的 Mono 图像上
- 每帧生成对比图: 左半 = H0 wide (参考标注来源) / 右半 = Hk mono (高度变换后)
- 标注叠加: lane_lines (实线) + road_edges (虚线) + 概率值文字
- 低置信度标注 (prob < 0.3) 用灰色半透明绘制

```python
def visualize_3d_annotation(
  session_dir: Path,
  height_tag: str,         # 'H1', 'H3', 'H6'
  frame_id: str,
) -> np.ndarray:
  """将 3D 标注投影到 Mono 图像并可视化。

  1. 读取 3d_labels/<height_tag>/<frame_id>.json
  2. 读取 Mono 图像 <height_tag>/<frame_id>.*
  3. 用 project_3d_to_mono(K_MONO, rpyCalib) 投影到 1920×1080
  4. 绘制 lane_lines + road_edges + prob 文字
  """
```

**颜色约定**:
- L-out (lane 0): 蓝色 / L-inn (lane 1): 绿色
- R-inn (lane 2): 红色 / R-out (lane 3): 黄色
- 左路沿 (RE 0): 青色虚线 / 右路沿 (RE 1): 品红虚线

**关键验证点**:
- 车道线是否贴合实际车道标线
- H1 (1.22m) vs H6 (3.00m) 在同一帧的标注差异是否合理
- 弯道处标注是否平滑无跳变
- 低概率标注（灰色）是否确实质量差

#### 4.8.3 `viz_clean.py` — Phase 3: 清洗与抽样统计

```bash
python tools/dashcam/tusimple/viz_clean.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --show-rejected 10
```

**检查内容**:

1. **统计摘要** (终端输出 + 可选图表):

```
质量过滤: 1500 帧 → 1260 通过 (84.0%)
  (500 unique frame × 3 heights = 1500 帧×高度对)
  拒绝原因:
    ll_prob < 0.3:  156 帧 (10.4%)
    v_ego < 1.0:    54 帧 (3.6%)
    both:            30 帧 (2.0%)
时间抽样: 1260 → 252 帧 (每 5 帧取 1)
划分: train=202 / val=26 / test=24

各高度分布:
  H1 (1.22m): train=67 / val=9 / test=8
  H3 (1.50m): train=67 / val=9 / test=8
  H6 (3.00m): train=68 / val=8 / test=8
```

2. **拒绝帧示例** (`--show-rejected N`):
   - 展示 N 个被拒绝帧的 Mono 图像 + 拒绝原因
   - 帮助判断过滤阈值是否合理（过严/过松）

3. **抽样分布图** (可选 `--plot`):
   - 帧 ID 时间轴上标注: 保留帧 (绿点) vs 拒绝帧 (红点)
   - 确认抽样是否均匀分散，无连续大段空白

#### 4.8.4 `viz_tusimple.py` — Phase 4: TuSimple 2D 标注检查

```bash
python tools/dashcam/tusimple/viz_tusimple.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --height H1 --max-frames 20
```

**检查内容**:

1. **TuSimple 标注叠加** (主要功能):
   - 在 1280×720 裁剪图上绘制 4 条车道线
   - 每个 h_sample 处画点，有效点间画连线
   - 补位来源标注: lane_line 来源画实心点，road_edge 补位画空心点

2. **ROI 裁剪对比** (`--show-crop`):
   - 左: 原始 1920×1080 图像，叠加裁剪框 (绿色矩形)
   - 右: 裁剪后 1280×720 图像 + TuSimple 标注
   - 验证裁剪区域是否合理覆盖道路

```
┌──────────────────────┬──────────────────┐
│ 原始 1920×1080       │ 裁剪后 1280×720  │
│ ┌──────────┐         │ ─── L-out (蓝)   │
│ │ ROI 裁剪  │ ←绿框  │ ─── L-inn (绿)   │
│ │ 776×436  │         │ ─── R-inn (红)   │
│ └──────────┘         │ ─── R-out (黄)   │
└──────────────────────┴──────────────────┘
```

3. **多高度对比** (`--compare-heights`):
   - 同一帧在 H1/H3/H6 的 TuSimple 标注并排显示
   - 验证: 近场差异小、远场差异大（高处相机远场下压）

```python
def visualize_tusimple_frame(
  image_path: Path,
  lanes: list[list[int]],
  h_samples: list[int],
  output_path: Path | None = None,
  show: bool = True,
) -> np.ndarray:
  """在 1280×720 图像上绘制 TuSimple 车道线标注。"""

def visualize_crop_comparison(
  mono_image_path: Path,
  tusimple_image_path: Path,
  crop_rect: tuple[int, int, int, int],
  lanes: list[list[int]],
  h_samples: list[int],
) -> np.ndarray:
  """原始图像 + 裁剪框 vs 裁剪后图像 + TuSimple 标注。"""
```

## 五、TuSimple 输出格式规范

### 5.1 JSON Lines 格式

每行一个 JSON 对象:

```json
{
  "lanes": [[-2, -2, 580, 560, 540, 520, ...], [-2, -2, 720, 700, 680, 660, ...]],
  "h_samples": [160, 170, 180, 190, 200, 210, ..., 710],
  "raw_file": "H1/images/000000.jpg"
}
```

### 5.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `lanes` | `list[list[int]]` | 车道线列表，固定 4 条: L-out, L-inn, R-inn, R-out (低置信度 lane_line 由 road_edge 补位) |
| `h_samples` | `list[int]` | 固定值 [160, 170, ..., 710]，共 56 个采样点 |
| `raw_file` | `str` | 对应图像文件的相对路径 |

### 5.3 车道线编码

- 每条车道线是一个长度 56 的整数列表，与 `h_samples` 一一对应
- `lanes[i][j]` = 第 i 条车道线在 `v = h_samples[j]` 处的 x 像素坐标
- `-2` 表示该采样点处无车道线（超出图像范围或不可见）
- 有效 x 范围: [0, 1279]

### 5.4 与标准 TuSimple 的兼容性

| 维度 | 标准 TuSimple | 本系统输出 |
|------|-------------|-----------|
| 图像分辨率 | 1280×720 | 1280×720 ✅ |
| h_samples | [160, 170, ..., 710] | [160, 170, ..., 710] ✅ |
| 车道线数量 | 通常 4~5 条 | 固定 4 条 (lane_line 优先, road_edge 补位) ✅ |
| JSON 格式 | JSON Lines | JSON Lines ✅ |
| 标注来源 | 人工标注 | 模型推理 + 3D 投影 |

## 六、关键复用的现有代码

| 功能 | 来源文件 | 具体函数/常量 |
|------|---------|-------------|
| Carla 相机管理 | `tools/dashcam/carla_multi_height_world.py` | 类结构模式、帧同步、回调机制 |
| 3D 标注推理 | `tools/dashcam/annotate_multi_height.py` | `decode_model_output()`, `transform_annotation()`, `_load_or_compile_model()`, `_run_inference()` |
| 投影数学 | `tools/dashcam/visualizer.py:49-65` | `project_points_to_image()` 投影公式 |
| 坐标变换 | `common/transformations/camera.py` | `view_frame_from_device_frame`, `CameraConfig`, `DEVICE_CAMERAS` |
| 旋转矩阵 | `common/transformations/orientation.py` | `rot_from_euler()` |
| 采集模式 | `tools/dashcam/collect_multi_height.py` | save_every、ThreadPoolExecutor 异步写、进度打印 |
| 预处理 | `tools/dashcam/train/dataset.py` | `rgb_to_modeld_input()` |
| GPU 预处理 | `tools/dashcam/modeld_preprocess_cl.py` | `ModeldInputPreprocessorCL` |
| 模型常量 | `selfdrive/modeld/constants.py` | `ModelConstants.X_IDXS`, `IDX_N` |

## 七、实现顺序 (依赖关系)

```
1. config.py              ← 无依赖，定义所有常量
2. projection.py          ← config + openpilot transformations
3. carla_world.py         ← config，参考 carla_multi_height_world 模式
4. collect.py             ← carla_world + config (Phase 1)
5. annotate_3d.py         ← config，复用 annotate_multi_height (Phase 2)
6. clean_and_sample.py    ← config (Phase 3)
7. project_tusimple.py    ← projection + config (Phase 4)
8. viz_collect.py         ← config (Phase 1 可视化)
9. viz_annotate.py        ← config + projection (Phase 2 可视化)
10. viz_clean.py          ← config (Phase 3 可视化)
11. viz_tusimple.py       ← config + projection (Phase 4 可视化)
```

## 八、验证方案

### 8.1 投影正确性验证

**方法**: 对 H1 高度的同一帧（H0 和 H1 同高度 1.22m）：
1. 用 `visualizer.py:project_points_to_image()` 投影到 H0 窄角相机 (K_narrow=2648)
2. 用 `projection.py:project_3d_to_mono()` 投影到 H1 Mono 相机 (K_MONO=554.3)
3. 在两个图像的重叠 FOV 区域，车道线位置应视觉一致

### 8.2 TuSimple 格式校验

编写验证函数:
```python
def validate_tusimple_labels(labels_path: Path) -> dict:
  """校验 TuSimple JSON Lines 格式。
  检查项:
    - 每行是合法 JSON
    - 含 'lanes', 'h_samples', 'raw_file' 字段
    - h_samples == TUSIMPLE_H_SAMPLES
    - 每条 lane 长度 == len(h_samples)
    - x 值为 int，-2 或 [0, 1279]
    - 图像文件存在且为 1280×720
  """
```

### 8.3 各 Phase 可视化检查

每个 Phase 完成后用对应的可视化脚本进行人工检查:

- **`viz_collect.py`**: 8 相机网格视图 — 帧同步、视角、场景是否正常
- **`viz_annotate.py`**: 3D 标注投影叠加 — 车道线贴合度、高度变换、弯道质量
- **`viz_clean.py`**: 清洗统计 — 通过率、拒绝帧示例、抽样分布
- **`viz_tusimple.py`**: TuSimple 2D 标注 — 车道线位置、ROI 裁剪覆盖、多高度对比

### 8.4 高度一致性验证

在直线道路段:
- H1 vs H6 在近场 (图像底部) 差异小
- H1 vs H6 在远场 (图像顶部) 差异大
- 这符合物理直觉：高处相机远场下压更明显

### 8.5 单元测试

构造已知的 3D 直线车道线，验证投影和插值结果:

```python
def test_straight_lane_projection():
  """直线车道线投影解析验证。
  构造: y=1.8m (右侧), z=1.22m (地面), x=5~100m
  验证: 投影后 u 应该在图像右半部分，v 应该从上到下递增
  """
```

### 8.6 弯道单调性裁断测试

```python
def test_monotonic_straight_lane():
  """直线车道线: v 单调递减，无截断，完整保留所有点。"""

def test_monotonic_curve_cutoff():
  """弯道车道线: v 在第 5 个点折返 (v[5] >= v[4])，
  截断后仅保留前 5 个点，后续 h_samples 返回 -2。"""

def test_monotonic_immediate_fold():
  """极端弯道: v[1] >= v[0]，仅保留 1 个点 → 不足 2 点 → 全 -2。"""

def test_monotonic_slight_curve():
  """缓弯: v 始终单调递减但间距不均匀，无截断，插值正确。"""
```

### 8.7 ROI 裁剪边界值测试

```python
def test_crop_params_nominal():
  """默认参数 (NOMINAL_PITCH=4°, crop_hfov=70°) 的裁剪区域和内参验证。"""

def test_crop_params_pitch_zero():
  """pitch=0° (水平) 时裁剪区域不越界，地平线在图像中部。"""

def test_crop_params_pitch_extreme():
  """pitch=10° (极端下俯) 时 crop_y 夹断到图像边界，不越界。"""

def test_crop_params_fov_range():
  """crop_hfov=50°/80°/120° 时裁剪宽度合理，内参一致。"""
```

### 8.8 车道线选择与 road_edge 补位测试

```python
def test_lane_selection_all_high_prob():
  """所有 lane_line prob > threshold → 直接使用 4 条 lane_line。"""

def test_lane_selection_road_edge_fallback():
  """lane[0] prob 低 + road_edge[0] 可用 → L-out 使用 RE[0] 补位。
  lane[3] prob 低 + road_edge[1] 可用 → R-out 使用 RE[1] 补位。"""

def test_lane_selection_no_fallback():
  """lane[0] prob 低 + road_edge[0] 也不可用 → L-out 全 -2。"""

def test_lane_selection_inner_no_fallback():
  """lane[1] prob 低 → L-inn 全 -2 (无 road_edge 补位源)。"""
```

### 8.9 幂等性验证

```python
def test_phase2_idempotent():
  """Phase 2 重复运行同一 session，不产生重复标注，跳过已存在文件。"""

def test_phase4_idempotent():
  """Phase 4 重复运行，已存在的 JPEG/JSON 不被覆盖。"""
```

## 九、完整使用流程

### Step 0: 小规模烟雾测试

首次运行建议先用 3 个高度 + 1 个场景 + 500 帧做端到端验证，确认存储/吞吐/投影质量后再扩展到全量：

```bash
DETACH=1 bash tools/dashcam/start_carla.sh
SESSION=data/tusimple/Town04_ClearNoon_p4.0_y0.0

# Phase 1: 少量帧采集
python tools/dashcam/tusimple/collect.py \
  --heights H1 H3 H6 \
  --map Town04 --weather ClearNoon \
  --pitch 4.0 --yaw 0.0 \
  --max-frames 500 --save-every 4 \
  --output-base data/tusimple \
  --mono-jpeg-quality 95
python tools/dashcam/tusimple/viz_collect.py $SESSION --max-frames 5

# Phase 2: 3D 标注
python tools/dashcam/tusimple/annotate_3d.py $SESSION --heights H1 H3 H6
python tools/dashcam/tusimple/viz_annotate.py $SESSION --heights H1 H3 H6 --max-frames 5

# Phase 3: 清洗抽样
python tools/dashcam/tusimple/clean_and_sample.py $SESSION --heights H1 H3 H6
python tools/dashcam/tusimple/viz_clean.py $SESSION --show-rejected 5

# Phase 4: TuSimple 投影
python tools/dashcam/tusimple/project_tusimple.py $SESSION
python tools/dashcam/tusimple/viz_tusimple.py $SESSION --height H1 --show-crop --max-frames 5
```

确认无误后再进入正式采集。

### Step 1: 启动 Carla

```bash
DETACH=1 bash tools/dashcam/start_carla.sh
```

### Step 2: Phase 1 — 数据采集

**方式 A: 单 session 采集** (调试/测试用)

```bash
python tools/dashcam/tusimple/collect.py \
  --heights H1 H2 H3 H4 H5 H6 \
  --map Town04 --weather ClearNoon \
  --pitch 4.0 --yaw 0.0 \
  --max-frames 500 --save-every 4 \
  --output-base data/tusimple \
  --mono-jpeg-quality 95
```

输出: `data/tusimple/Town04_ClearNoon_p4.0_y0.0/` (H0/ + H1~H6/)

**方式 B: 批量采集** (正式训练数据，全场景×姿态矩阵)

```bash
# 查看采集计划
python tools/dashcam/tusimple/run_full_collection.py --list

# 启动批量采集 (136 session, 断点可恢复)
python tools/dashcam/tusimple/run_full_collection.py \
  --output-base data/tusimple \
  --max-frames 500 --save-every 4 \
  --num-npc 40 --no-display \
  --mono-jpeg-quality 95
```

输出: `data/tusimple/` 下 136 个 session 目录，每个含 H0/ + H1~H6/

**Phase 1 可视化检查**:

```bash
python tools/dashcam/tusimple/viz_collect.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --max-frames 10
```

确认: 8 相机网格视图正常、帧同步、各高度视角差异合理。

### Step 3: Phase 2 — 3D 标注

```bash
python tools/dashcam/tusimple/annotate_3d.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --onnx selfdrive/modeld/models/driving_vision.onnx \
  --heights H1 H3 H6 \
  --min-ll-prob 0.3
```

输出: `data/tusimple/Town04_ClearNoon_p4.0_y0.0/3d_labels/` (H1/ H3/ H6/)

**Phase 2 可视化检查**:

```bash
python tools/dashcam/tusimple/viz_annotate.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --heights H1 H3 H6 --max-frames 10
```

确认: 3D 车道线投影贴合路面、高度变换后标注合理、弯道处无跳变。

### Step 4: Phase 3 — 数据清洗与抽样

```bash
python tools/dashcam/tusimple/clean_and_sample.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --heights H1 H3 H6 \
  --min-ll-prob 0.3 --min-speed 1.0 \
  --sample-every 5 \
  --split-ratio 0.8 0.1 0.1 --seed 42
```

输出: `data/tusimple/Town04_ClearNoon_p4.0_y0.0/splits/` (train.txt, val.txt, test.txt)

**Phase 3 可视化检查**:

```bash
python tools/dashcam/tusimple/viz_clean.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --show-rejected 10
```

确认: 通过率合理 (>80%)、拒绝帧确实质量差、抽样分布均匀。

### Step 5: Phase 4 — TuSimple 投影

```bash
python tools/dashcam/tusimple/project_tusimple.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --crop-hfov 70
```

输出: `data/tusimple/Town04_ClearNoon_p4.0_y0.0/tusimple/` (train.json, val.json, test.json + images/)
注: 使用固定裁剪 (NOMINAL_PITCH=4°)，图像从 1920×1080 (120°) ROI 裁剪到 ~70° 后缩放为 1280×720

**Phase 4 可视化检查**:

```bash
# TuSimple 标注叠加
python tools/dashcam/tusimple/viz_tusimple.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --height H1 --max-frames 20

# ROI 裁剪对比 (原始 vs 裁剪后)
python tools/dashcam/tusimple/viz_tusimple.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --height H1 --show-crop --max-frames 5

# 多高度并排对比
python tools/dashcam/tusimple/viz_tusimple.py \
  data/tusimple/Town04_ClearNoon_p4.0_y0.0/ \
  --compare-heights --max-frames 5
```

确认: 车道线贴合路面、ROI 裁剪覆盖完整道路区域、不同高度标注差异符合物理预期。

## 十、已知限制与后续工作

1. **针孔 vs 鱼眼**: 当前使用 Carla 针孔模型近似 120° 鱼眼。如需更高保真度，可在后处理中添加合成鱼眼畸变
2. **标注来源**: 3D 标注来自 openpilot 预训练模型 (H0) 推理，非真实地面真值。标注精度受限于模型性能
3. **弯道处理**: 大曲率弯道通过单调性裁断在折返点截断车道线，保证投影正确性。代价是弯道处车道线比直线时短
4. **场景多样性**: 需要在多种地图、天气、pitch/yaw 组合下采集，才能获得鲁棒的训练数据
5. **road_edges 作为补位源**: 当外侧 lane_line 置信度低时用 road_edge 补位。补位后的车道线在 TuSimple 格式中与普通 lane_line 无区分标记
6. **ROI 裁剪 FOV 选择**: 默认 70° 裁剪是远场分辨率与视野宽度的折中。极窄道路或多车道场景可能需要更大 FOV (可通过 `--crop-hfov` 调整)
7. **固定裁剪区域**: 使用 NOMINAL_PITCH=4° 计算固定裁剪位置，训练和部署共用。训练数据覆盖 0°~7° pitch 范围确保鲁棒性。若部署 pitch 超出此范围需重新评估
