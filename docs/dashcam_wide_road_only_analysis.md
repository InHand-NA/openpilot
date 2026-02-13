# Dashcam Wide-Road-Only 模式感知效果分析与提升方案

## 背景

dashcam 工具支持三种相机模式：

| 参数 | VisionIPC 流 | modeld 行为 |
|------|-------------|------------|
| （默认双目） | ROAD + WIDE_ROAD | 双目，fcam + ecam intrinsics |
| `--wide-road-only` | WIDE_ROAD only | 单目，ecam intrinsics for both inputs |
| `--road-only` | ROAD only | 单目，fcam intrinsics for main input |

实测发现 **wide-road-only 模式的感知效果明显差于 road-only 和双目模式**。本文分析根因并提出改进方案。

## 相机与模型参数

### 相机 intrinsics

| 相机 | 焦距 (px) | 分辨率 | 水平 FOV | Carla FOV 设置 |
|------|----------|--------|---------|---------------|
| fcam (窄角) | 2648 | 1928×1208 | ~40° | `fov=40` |
| ecam (广角) | 567 | 1928×1208 | ~119° | `fov=120` |

定义位置：`common/transformations/camera.py:49-52`

```python
_ar_ox_fisheye = CameraConfig(1928, 1208, 567.0)   # ecam
_ar_ox_config = DeviceCameraConfig(
  CameraConfig(1928, 1208, 2648.0),  # fcam
  _ar_ox_fisheye,                     # dcam
  _ar_ox_fisheye)                     # ecam
```

### 模型输入参数

modeld 神经网络有两个图像输入：

| 输入名 | 模型 | 输入尺寸 | 目标焦距 | 水平 FOV | 用途 |
|--------|------|---------|----------|---------|------|
| `img` | MEDMODEL | 512×256 | 910 | ~31° | 主感知流（车道线、路边沿、前车） |
| `big_img` | SBIGMODEL | 512×256 | 455 | ~59° | 辅助广角感知流 |

定义位置：`common/transformations/model.py:9-40`

```python
# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
medmodel_fl = 910.0

# SBIG model
SBIGMODEL_INPUT_SIZE = (512, 256)
sbigmodel_fl = 455.0
```

## Warp 变换机制

modeld 通过 `get_warp_matrix()` 计算从相机原图到模型输入的透视变换矩阵：

```python
# common/transformations/model.py:65-70
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix
```

**缩放关系**：`f_camera / f_model` 决定了从原图采样多少像素送入模型。
- `f_camera > f_model` → **降采样**（多个原始像素合并为一个模型像素，信息丰富）
- `f_camera < f_model` → **上采样**（一个原始像素插值为多个模型像素，信息不足）

## 根因分析

### 三种模式下的 MEDMODEL（主输入 `img`）采样对比

| 模式 | 相机焦距 | 模型焦距 | 比值 | 原图采样区域 | 采样方式 | 结果 |
|------|---------|---------|------|------------|---------|------|
| road-only | fcam 2648 | 910 | 2.91 | 中心 1490px (77%) | **2.91x 降采样** | 锐利 |
| dual (默认) | fcam 2648 | 910 | 2.91 | 中心 1490px (77%) | **2.91x 降采样** | 锐利 |
| **wide-road-only** | ecam 567 | 910 | 0.623 | 中心 319px (16.5%) | **1.60x 上采样** | 模糊 |

### 三种模式下的 SBIGMODEL（辅助输入 `big_img`）采样对比

| 模式 | 相机焦距 | 模型焦距 | 比值 | 原图采样区域 | 采样方式 | 结果 |
|------|---------|---------|------|------------|---------|------|
| road-only | fcam 2648* | 455 | 5.82 | 中心极小区域 | 5.82x 降采样 | 过度裁剪 |
| dual (默认) | ecam 567 | 455 | 1.25 | 中心 638px (33%) | 1.25x 降采样 | 良好 |
| **wide-road-only** | ecam 567 | 455 | 1.25 | 中心 638px (33%) | **1.25x 降采样** | 良好 |

> *road-only 模式下 `buf_extra = buf_main`（同一张窄角图），但 `model_transform_extra` 仍使用 ecam intrinsics

### 结论

**wide-road-only 感知差的根本原因**：MEDMODEL（主感知输入）从广角图的中心仅 319 个像素做 1.60x 上采样到 512 像素，像素信息量严重不足。

对比：road-only 模式下 MEDMODEL 从 1490 个像素做 2.91x 降采样，像素信息量是 wide-road-only 的 **4.67 倍**。

SBIGMODEL（辅助输入）在 wide-road-only 下表现良好（1.25x 降采样），但无法完全弥补主输入的退化。

### 直观示意

```
广角原图 (1928 px wide, FOV≈120°)
┌──────────────────────────────────────────────────────────┐
│                                                          │
│             ┌──── 319px ────┐                            │
│             │   MEDMODEL    │      SBIGMODEL             │
│             │  采样区域     │      采样区域              │
│             │  (上采样1.6x) │    ┌── 638px ──┐           │
│             │  ❌ 模糊      │    │ (降采样1.25x)          │
│             └───────────────┘    │ ✅ 清晰    │           │
│                                  └────────────┘           │
└──────────────────────────────────────────────────────────┘

窄角原图 (1928 px wide, FOV≈40°)
┌──────────────────────────────────────────────────────────┐
│                                                          │
│    ┌──────────────── 1490px ────────────────┐            │
│    │            MEDMODEL 采样区域            │            │
│    │          (降采样 2.91x) ✅ 锐利         │            │
│    └────────────────────────────────────────┘            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## modeld 相关代码路径

### 流检测与模式判断 (`selfdrive/modeld/modeld.py:330-343`)

```python
available_streams = VisionIpcClient.available_streams("camerad", block=False)
use_extra_client = VISION_STREAM_WIDE_ROAD in available_streams and VISION_STREAM_ROAD in available_streams
main_wide_camera = VISION_STREAM_ROAD not in available_streams

vipc_client_main_stream = VISION_STREAM_WIDE_ROAD if main_wide_camera else VISION_STREAM_ROAD
```

### 变换矩阵计算 (`selfdrive/modeld/modeld.py:429-436`)

```python
dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
# 主流：wide-road-only 时用 ecam intrinsics
model_transform_main = get_warp_matrix(euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False)
# 副流：始终用 ecam intrinsics + SBIGMODEL warp
model_transform_extra = get_warp_matrix(euler, dc.ecam.intrinsics, True)
```

### 单目 fallback (`selfdrive/modeld/modeld.py:416-420`)

```python
else:
    # Use single camera
    buf_extra = buf_main
    meta_extra = meta_main
```

### 输入分配 (`selfdrive/modeld/modeld.py:460-461`)

```python
bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
```

## 提升方案

### 方案 A：虚拟双目——从单个广角 Carla 相机合成双路流（推荐）

**思路**：只用一个 Carla 广角相机（物理单目），但以更高分辨率渲染，在软件层合成两路 VisionIPC 流：

- **WIDE_ROAD**：全幅降采样到 1928×1208
- **ROAD**：裁剪中心 40° FOV 区域，缩放到 1928×1208

modeld 看到双路流 → `use_extra_client=True`，`main_wide_camera=False` → 正常双目模式，两个网络输入都能获得足够像素信息。

**中心裁剪计算**（FOV=120° → 裁剪 FOV=40° 中心区域）：

```
裁剪比例 = tan(20°) / tan(60°) = 0.364 / 1.732 = 0.210
```

| Carla 渲染分辨率 | 中心 40° 裁剪尺寸 | 上采样到 1928 的倍数 | MEDMODEL 有效采样 |
|-----------------|------------------|-------------------|-----------------|
| 1928×1208 (1x) | 405×254 | 4.76x | 0.61x 上采样 ❌ |
| 3856×2416 (2x) | 810×508 | 2.38x | 1.22x 降采样 ✅ |
| 5784×3624 (3x) | 1216×760 | 1.59x | 1.83x 降采样 ✅ |

**推荐起步分辨率**：2x (3856×2416)，MEDMODEL 有效降采样 1.22x，效果接近真双目。

**改动范围**：`camerad.py`（合成双路流）、`carla_world.py`（高分辨率渲染 + 裁剪/缩放）

**优点**：
- 单个 Carla 相机 → modeld 完整双目模式
- 两个网络输入都获得足够像素信息
- 对 modeld 完全透明，无需修改模型代码

**缺点**：
- Carla 需要渲染 2-3x 分辨率，GPU 开销增加（约 4-9x 像素量）
- 虚拟窄角流有一定插值损失（2x 分辨率下约 2.4x 上采样）

### 方案 B：保持 wide-road-only，Carla 高分辨率渲染再降采样

**思路**：仍然只发 WIDE_ROAD 流（modeld 保持单目模式），但让 Carla 广角相机以更高分辨率渲染，降采样到 1928×1208 后再发送。

**效果**：降采样引入抗锯齿，边缘更平滑，但 MEDMODEL 的根本问题（319px→512px 上采样）不变。提升有限。

**优点**：实现简单，仅改 `carla_world.py`
**缺点**：无法解决根本问题

### 方案 C：修改 modeld warp 逻辑

**思路**：fork modeld 到 `tools/dashcam/modeld.py`，当只有 WIDE_ROAD 时，让 MEDMODEL 也使用 SBIGMODEL 的 warp（567→455 降采样）而非 MEDMODEL 原始 warp（567→910 上采样）：

```python
# 修改后（wide-road-only 时）：
model_transform_main = get_warp_matrix(euler, ecam.intrinsics, True)   # 改用 SBIGMODEL warp
model_transform_extra = get_warp_matrix(euler, ecam.intrinsics, True)  # 不变
```

**优点**：零额外 GPU 开销
**缺点**：MEDMODEL 的裁剪区域不再匹配训练时的几何分布，模型可能输出异常（需实验验证）

### 方案对比

| 方案 | 改动范围 | GPU 开销 | 预期效果 | 风险 |
|------|---------|---------|---------|------|
| **A. 虚拟双目** | camerad + carla_world | 增加（2-3x 渲染） | **好**（等同真双目） | 低 |
| B. 高分辨率降采样 | carla_world | 略增（2x 渲染） | 略有改善 | 低 |
| C. 修改 modeld warp | fork modeld | 不变 | 不确定 | 高 |

**推荐方案 A**：虚拟双目是最可靠的方案，modeld 进入完整双目模式，MEDMODEL 和 SBIGMODEL 两个输入都能获得足够的像素信息，且不需要修改模型代码。

## 参考文件

| 文件路径 | 内容 |
|---------|------|
| `common/transformations/model.py` | MEDMODEL/SBIGMODEL 参数、`get_warp_matrix()` |
| `common/transformations/camera.py` | fcam/ecam intrinsics、DEVICE_CAMERAS |
| `selfdrive/modeld/modeld.py` | 流检测、warp 计算、单目 fallback |
| `selfdrive/modeld/models/commonmodel.h` | MODEL_WIDTH/HEIGHT 常量 |
| `selfdrive/modeld/transforms/transform.cc` | OpenCL warp 核心实现 |
| `tools/dashcam/camerad.py` | VisionIPC 流创建 |
| `tools/dashcam/carla_world.py` | Carla 相机创建与帧采集 |
| `tools/dashcam/run.py` | 主循环、模式选择 |
