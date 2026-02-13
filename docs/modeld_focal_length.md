# modeld 模型焦距（Model Focal Length）概念解析

## 背景

openpilot 的感知模型 modeld 接收相机原图（1928×1208），通过 GPU warp 变换裁剪/缩放到模型输入尺寸（512×256），再送入神经网络推理。这个 warp 变换的核心参数之一就是**模型焦距**。

本文解释模型焦距的含义、它与相机焦距的关系、以及它如何影响感知质量。

## 从 3D 世界到模型输入像素的完整变换链

modeld 通过 `get_warp_matrix()` 计算从模型输入像素到相机原图像素的映射矩阵（`common/transformations/model.py:65-70`）：

```python
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix
```

变换链经过 4 步坐标转换：

```
模型像素 (u_m, v_m)
    ↓  inv(K_model)          — 模型像素 → 归一化 3D 射线（模型虚拟相机坐标系）
标定坐标系 3D 射线
    ↓  device_from_calib      — 标定 → 设备坐标系（补偿相机安装偏差 roll/pitch/yaw）
设备坐标系 3D 射线
    ↓  view_from_device       — 设备 → 视图坐标系（x 右, y 下, z 前）
视图坐标系 3D 射线
    ↓  K_camera               — 归一化 3D 射线 → 相机像素 (u_c, v_c)
相机像素 (u_c, v_c)
```

GPU 的 `warpPerspective` 内核（`selfdrive/modeld/transforms/transform.cl`）对模型输出的每个像素 `(u_m, v_m)` 执行此变换，找到对应的相机原图像素 `(u_c, v_c)`，做双线性插值采样。

## 模型焦距的定义

**模型焦距是模型输入图像的「虚拟相机」内参**。它不是物理镜头的属性，而是在模型训练时固定下来的几何约束，定义了模型输入每个像素对应的 3D 视角。

```python
# common/transformations/model.py:9-18
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_CY = 47.6

medmodel_fl = 910.0                        # MEDMODEL 焦距
medmodel_intrinsics = np.array([
  [910.0,   0.0,  256.0],                  # fx, 0, cx (cx = 512/2)
  [  0.0, 910.0,   47.6],                  # 0, fy, cy (偏上，因为道路在图像下半部分)
  [  0.0,   0.0,    1.0]])
```

```python
# common/transformations/model.py:32-40
SBIGMODEL_INPUT_SIZE = (512, 256)

sbigmodel_fl = 455.0                        # SBIGMODEL 焦距
sbigmodel_intrinsics = np.array([
  [455.0,   0.0,  256.0],                  # fx, 0, cx
  [  0.0, 455.0,  151.8],                  # 0, fy, cy = 0.5*(256+47.6)
  [  0.0,   0.0,    1.0]])
```

### 模型焦距决定视野范围

```
模型水平 FOV = 2 × atan(width / 2 / f_model)
```

| 模型 | 输入尺寸 | 焦距 | 水平 FOV | 用途 |
|------|---------|------|---------|------|
| MEDMODEL | 512×256 | 910 | ~31° | 主感知流（车道线、路边沿、前车） |
| SBIGMODEL | 512×256 | 455 | ~59° | 辅助广角感知流 |

MEDMODEL 看窄（31°，集中在正前方），SBIGMODEL 看宽（59°，覆盖两侧车道）。两者尺寸相同但焦距不同，分工互补。

### 光心偏移 (cy) 的含义

MEDMODEL 的 `cy = 47.6`（远小于图像中心 128），说明模型的光心在图像上方，画面中更多区域分配给了路面和前方车辆（图像下半部分），天空占比很小。这是在训练时根据驾驶场景优化的。

## 关键比值：f_camera / f_model

当标定为零（理想相机安装，无 pitch/yaw/roll 偏差）时，warp 矩阵简化为：

```
warp ≈ K_camera @ inv(K_model)
```

对于中心附近像素，变换关系简化为线性缩放：

```
u_camera = (f_camera / f_model) × (u_model - cx_model) + cx_camera
```

**`f_camera / f_model` 就是从模型像素到相机像素的缩放比**，它决定了感知质量：

- **比值 > 1（降采样）**：多个相机像素合并为一个模型像素，信息充足，图像锐利
- **比值 = 1**：一对一映射，既不丢失也不插值
- **比值 < 1（上采样）**：一个相机像素拉伸为多个模型像素，信息不足，图像模糊

### 不同相机 × 模型组合的采样效果

相机参数（`common/transformations/camera.py:49-51`）：

| 相机 | 焦距 (px) | 分辨率 | 水平 FOV | Carla FOV 设置 |
|------|----------|--------|---------|---------------|
| fcam（窄角） | 2648 | 1928×1208 | ~40° | `fov=40` |
| ecam（广角） | 567 | 1928×1208 | ~119° | `fov=120` |

采样分析：

| 组合 | f_camera | f_model | 比值 | 相机原图采样宽度 | 采样方式 | 效果 |
|------|---------|---------|------|----------------|---------|------|
| MEDMODEL + fcam | 2648 | 910 | **2.91** | 1490px (77%) | 2.91x 降采样 | 锐利 ✅ |
| MEDMODEL + ecam | 567 | 910 | **0.62** | 319px (17%) | 1.60x 上采样 | 模糊 ❌ |
| SBIGMODEL + ecam | 567 | 455 | **1.25** | 638px (33%) | 1.25x 降采样 | 良好 ✅ |
| SBIGMODEL + fcam | 2648 | 455 | **5.82** | 极小区域 | 5.82x 降采样 | 过度裁剪 ⚠️ |

> 采样宽度 = f_camera / f_model × model_width = f_camera / f_model × 512

## 几何直觉

模型焦距 910 意味着模型「期望」在中心像素两侧 256 像素处看到 ±atan(256/910) ≈ ±15.7° 的场景。

- 用 **fcam (f=2648)** 喂给 MEDMODEL：fcam 在 ±15.7° 范围内有 2648×tan(15.7°)×2 ≈ 1490 像素的信息。从 1490 像素降采样到 512 像素，**每个模型像素由 2.91 个真实像素支撑**。
- 用 **ecam (f=567)** 喂给 MEDMODEL：ecam 在同样 ±15.7° 范围内只有 567×tan(15.7°)×2 ≈ 319 像素的信息。从 319 像素上采样到 512 像素，**每个模型像素只有 0.62 个真实像素支撑**。

```
fcam 原图 (1928px, f=2648, FOV≈40°)
┌──────────────────────────────────────────────────────┐
│         ┌──────── 1490px (±15.7°) ────────┐          │
│         │  MEDMODEL 采样区域 → 512px       │          │
│         │  2.91 真实像素/模型像素  ✅ 锐利  │          │
│         └─────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘

ecam 原图 (1928px, f=567, FOV≈119°)
┌──────────────────────────────────────────────────────┐
│                    ┌── 319px (±15.7°) ──┐            │
│                    │ MEDMODEL 采样区域    │            │
│                    │ 319→512 上采样       │            │
│                    │ 0.62 真实像素  ❌ 糊 │            │
│                    └────────────────────┘            │
│                                                      │
│            ┌───── 638px (±35°) ─────┐                │
│            │ SBIGMODEL 采样区域      │                │
│            │ 1.25x 降采样  ✅ 良好   │                │
│            └─────────────────────────┘                │
└──────────────────────────────────────────────────────┘
```

## modeld 中的实际代码路径

### 流检测与模式判断

`selfdrive/modeld/modeld.py:330-343`：

```python
available_streams = VisionIpcClient.available_streams("camerad", block=False)
use_extra_client = VISION_STREAM_WIDE_ROAD in available_streams and VISION_STREAM_ROAD in available_streams
main_wide_camera = VISION_STREAM_ROAD not in available_streams
```

- **双目模式**：ROAD + WIDE_ROAD 都在 → `use_extra_client=True, main_wide_camera=False`
- **wide-road-only**：只有 WIDE_ROAD → `use_extra_client=False, main_wide_camera=True`
- **road-only**：只有 ROAD → `use_extra_client=False, main_wide_camera=False`

### 变换矩阵计算

`selfdrive/modeld/modeld.py:429-436`：

```python
dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
# 主流：wide-road-only 时用 ecam.intrinsics，否则用 fcam.intrinsics
model_transform_main = get_warp_matrix(euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False)
# 副流：始终用 ecam.intrinsics + SBIGMODEL warp
model_transform_extra = get_warp_matrix(euler, dc.ecam.intrinsics, True)
```

### 输入分配

`selfdrive/modeld/modeld.py:460-461`：

```python
bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
```

- 名称含 `big` 的输入 → 使用 extra 流 + SBIGMODEL warp
- 其他输入 → 使用 main 流 + MEDMODEL warp

### GPU warp 执行

`selfdrive/modeld/transforms/transform.cl` 中的 `warpPerspective` 内核：

```opencl
// 对模型输出的每个像素 (dx, dy)
float X0 = M[0] * dx + M[1] * dy + M[2];  // 齐次坐标 x
float Y0 = M[3] * dx + M[4] * dy + M[5];  // 齐次坐标 y
float W  = M[6] * dx + M[7] * dy + M[8];  // 齐次坐标 w
// 透视除法 → 源图像坐标
// 双线性插值采样 4 个相邻像素
```

## 总结

**模型焦距 = 训练时冻结的视角分配方案**。它规定了模型输入 512×256 图像中，每个像素应该对应真实世界的多大角度。推理时，`get_warp_matrix()` 根据实际相机焦距和模型焦距的比值，从相机原图中裁剪/缩放出正确的区域。当相机焦距远大于模型焦距时（如 fcam→MEDMODEL），裁剪区域大、降采样、信息充足；当相机焦距小于模型焦距时（如 ecam→MEDMODEL），裁剪区域小、上采样、信息不足——这就是 wide-road-only 模式感知效果差的根本原因。

## 参考文件

| 文件路径 | 内容 |
|---------|------|
| `common/transformations/model.py` | MEDMODEL/SBIGMODEL 参数、`get_warp_matrix()` |
| `common/transformations/camera.py` | fcam/ecam intrinsics、坐标系定义 |
| `selfdrive/modeld/modeld.py` | 流检测、warp 计算、输入分配 |
| `selfdrive/modeld/transforms/transform.cl` | OpenCL warp 内核（双线性插值） |
| `selfdrive/modeld/transforms/transform.cc` | warp 初始化、Y/UV 分别变换 |
| `selfdrive/modeld/models/commonmodel.h` | MODEL_WIDTH/HEIGHT 常量 |
