# 120° HFOV 鱼眼相机用于 openpilot Driving Vision 模型的可行性分析

## 1. 问题背景

### 1.1 目标

使用一颗 **HFOV 120° 的鱼眼相机**，接入 openpilot 的 driving vision 神经网络模型，实现：
- **LDW（Lane Departure Warning，车道偏离预警）**：依赖模型输出的 lane_lines 和 road_edges
- **FCW（Forward Collision Warning，前碰撞预警）**：依赖模型输出的 lead（前车检测）

### 1.2 挑战概述

openpilot 的图像处理流水线基于**针孔相机模型**，整个变换链使用 3×3 单应性矩阵（homography），**无法表达鱼眼镜头的非线性径向畸变**。120° HFOV 的鱼眼相机畸变显著，必须在进入 openpilot 流水线之前进行处理。

---

## 2. openpilot 图像处理流水线关键要点

> 完整流水线详见 [image_processing_pipeline.md](image_processing_pipeline.md)

### 2.1 模型输入规格

driving vision 模型有两个图像输入，分别对应两个**虚拟相机**：

| 输入名 | 虚拟相机 | 尺寸 | 焦距 (px) | cx | cy | 水平 FOV | 用途 |
|--------|---------|------|----------|----|----|---------|------|
| `img` | MEDMODEL | 512×256 | **910** | 256.0 | 47.6 | ~31° | 主感知：车道线、路边沿、前车 |
| `big_img` | SBIGMODEL | 512×256 | **455** | 256.0 | 151.8 | ~59° | 辅助广角感知 |

> 定义位置：`common/transformations/model.py:9-40`

### 2.2 Warp 变换

模型从物理相机原图中采样的核心机制是 `get_warp_matrix()`（`model.py:65-70`）：

```
warp_matrix = K_camera @ view_from_device @ device_from_calib @ calib_from_model
```

- `K_camera`：物理相机内参（针孔模型，仅 focal_length 和主点）
- `calib_from_model`：预计算的模型虚拟相机逆矩阵
- 这是一个 **3×3 单应性矩阵**，只能表达线性投影变换

### 2.3 关键约束

从 [camera_selection_guide.md](camera_selection_guide.md) 总结：

1. **MEDMODEL 不上采样**：`f_camera ≥ 910`（否则模型输入像素由少于一个物理像素支撑，图像模糊）
2. **SBIGMODEL FOV 覆盖**：`相机 HFOV ≥ 59°`（否则边缘采样到图像外）
3. **针孔模型假设**：整个流水线不含畸变校正，输入必须是（近似的）针孔相机图像

---

## 3. 鱼眼相机的核心挑战

### 3.1 投影模型差异

| 模型 | 投影公式 | 特征 |
|------|---------|------|
| 针孔 (pinhole) | r = f × tan(θ) | 边缘像素密度随角度急剧增大 |
| 等距鱼眼 (equidistant) | r = f × θ | 像素密度均匀分布 |
| OpenCV 鱼眼 | r = f × θ(1 + k₁θ² + ...) | 接近等距，含高阶修正 |

θ = 60°（120° HFOV 的边缘）处投影偏差：

| 模型 | r/f 值 | 相对针孔偏差 |
|------|--------|------------|
| 针孔 | tan(60°) = **1.732** | 基准 |
| 等距 | π/3 = **1.047** | **-39.5%** |

→ **不做去畸变处理，边缘区域的车道线位置偏差达数百像素**，完全无法使用。

### 3.2 为什么 openpilot 的 warp 无法处理鱼眼畸变

openpilot 的 OpenCL warp 内核执行的是**透视变换**（3×3 矩阵乘法），本质上是一个线性映射（齐次坐标下）。鱼眼畸变是**非线性的径向函数**，无法用任何 3×3 矩阵表达。

```
透视变换:  p_out = M × p_in      (线性，3×3 矩阵)
鱼眼畸变:  r = f(θ)               (非线性，r = √(x² + y²)，θ = arctan(r/f))
```

因此，**鱼眼相机图像不能直接送入 openpilot 流水线**，必须先进行去畸变处理。

---

## 4. 有效分辨率分析

这是可行性的核心判断依据。鱼眼相机的像素分辨率在去畸变后是否足够支撑模型输入？

### 4.1 分析方法

> 以下以理想等距 (equidistant) 模型为基础分析，实际鱼眼（OpenCV Kannala-Brandt 模型）与等距接近。

对于 HFOV = 120° 的等距鱼眼，像素焦距为：

```
f_fish = (W/2) / (HFOV/2) = (W/2) / (π/3) = 3W / (2π) ≈ 0.477 × W
```

鱼眼的角分辨率（每弧度对应的像素数）在全视场内**均匀**：

```
Δθ_fish = 1 / f_fish     (每像素覆盖的角度，常数)
```

模型虚拟相机在角度 θ 处每个像素覆盖的角度为：

```
Δθ_model = cos²(θ) / f_model    (针孔模型，θ 越大，每像素覆盖角度越小)
```

**有效分辨率比值**（源像素 / 模型像素，>1 为降采样即充足，<1 为上采样即不足）：

```
R(θ) = Δθ_model / Δθ_fish = f_fish × cos²(θ) / f_model
```

关键观察：**这个比值与中间步骤（去畸变图像）的参数无关**，只取决于鱼眼源分辨率和模型焦距。

### 4.2 各分辨率的分辨率比值

#### MEDMODEL（f_model = 910，半视场角 15.7°）

| 鱼眼分辨率 | f_fish (px/rad) | 中心 R(0°) | 边缘 R(15.7°) | 评估 |
|-----------|----------------|-----------|--------------|------|
| **1280×720** | 611 | 0.67 | 0.62 | ❌ 全域上采样，不可用 |
| **1920×1080** | 917 | **1.01** | **0.93** | ⚠️ 中心刚好，边缘略不足 |
| **2560×1440** | 1222 | **1.34** | **1.24** | ✅ 全域降采样，良好 |
| **3840×2160** | 1833 | **2.01** | **1.87** | ✅✅ 充足 |

#### SBIGMODEL（f_model = 455，半视场角 29.4°）

| 鱼眼分辨率 | f_fish (px/rad) | 中心 R(0°) | 边缘 R(29.4°) | 评估 |
|-----------|----------------|-----------|---------------|------|
| **1280×720** | 611 | 1.34 | 1.02 | ⚠️ 边缘勉强 |
| **1920×1080** | 917 | **2.02** | **1.53** | ✅ 良好 |
| **2560×1440** | 1222 | **2.69** | **2.04** | ✅✅ 优秀 |
| **3840×2160** | 1833 | **4.03** | **3.05** | ✅✅ 优秀 |

### 4.3 综合评估

| 鱼眼分辨率 | MEDMODEL | SBIGMODEL | 综合可行性 |
|-----------|----------|-----------|-----------|
| 1280×720 | ❌ 不可用 | ⚠️ 勉强 | **不可行** |
| 1920×1080 | ⚠️ 边缘可用 | ✅ 良好 | **勉强可行（MEDMODEL 边缘轻微模糊）** |
| 2560×1440 | ✅ 良好 | ✅✅ 优秀 | **可行，推荐** |
| 3840×2160 | ✅✅ 优秀 | ✅✅ 优秀 | **完全可行** |

---

## 5. 实现方案

### 5.1 总体思路

由于 openpilot warp 流水线假设针孔相机，必须在进入流水线之前将鱼眼图像转换为等效的针孔图像。有两种实现路径：

### 5.2 方案 A：鱼眼去畸变 + openpilot warp 流水线（两步法）

```
鱼眼原图 (W×H, fisheye, HFOV=120°)
      │
      │  cv2.remap (预计算查找表, 一次性标定)
      ↓
去畸变针孔图 (W_out×H_out, pinhole, HFOV=70°~90°)
      │
      │  VisionIPC 共享内存
      ↓
openpilot modeld (标准 warp 流水线)
      │  get_warp_matrix() 使用去畸变图的内参
      ↓
模型输入 (1, 12, 128, 256) × 2
```

#### 去畸变图像参数选择

去畸变后的针孔图像需要满足：
- `f_undistort ≥ 910`（MEDMODEL 不上采样）
- `HFOV_undistort ≥ 59°`（SBIGMODEL 完整覆盖）

但也不能选太高的分辨率（超过鱼眼源能提供的信息量）。**最优策略**：选择 f_undistort ≈ f_fish，使去畸变步骤在中心附近不上采样。

| 鱼眼分辨率 | f_fish | 推荐去畸变参数 | 去畸变分辨率 | f_undistort | HFOV |
|-----------|--------|-------------|------------|------------|------|
| 1920×1080 | 917 | 匹配 f_fish | 1290×730 | ~917 | ~70° |
| 2560×1440 | 1222 | 匹配 f_fish | 1712×970 | ~1222 | ~70° |
| 3840×2160 | 1833 | 匹配 f_fish | 2568×1452 | ~1833 | ~70° |

> 备选：如果计算资源有限，可以选择更小的去畸变图。例如 1920×1080 鱼眼 → 1024×576 去畸变图（HFOV ~60°, f ≈ 886），但 MEDMODEL 会有约 3% 的上采样。

#### 实现代码

```python
import cv2
import numpy as np

# ===== Step 1: 鱼眼标定（离线，一次性）=====
# 使用棋盘格标定获取:
K_fisheye = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]])  # 鱼眼内参
D = np.array([k1, k2, k3, k4])       # OpenCV fisheye 畸变系数

# ===== Step 2: 定义目标针孔内参 =====
# 目标: HFOV ~70°, 使 f_undistort ≈ f_fish
W_out, H_out = 1290, 730  # 示例（1920×1080 鱼眼）
f_undistort = W_out / (2 * np.tan(np.radians(70 / 2)))  # ≈ 921

K_undistort = np.array([
    [f_undistort, 0, W_out / 2],
    [0, f_undistort, H_out / 2],
    [0, 0, 1]
])

# ===== Step 3: 预计算 remap 查找表（离线，一次性）=====
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K=K_fisheye,
    D=D,
    R=np.eye(3),           # 如需 roll/pitch 补偿可加旋转
    P=K_undistort,
    size=(W_out, H_out),
    m1type=cv2.CV_16SC2,
)

# ===== Step 4: 运行时每帧处理 =====
frame_undistorted = cv2.remap(frame_fisheye, map1, map2, cv2.INTER_LINEAR)
# frame_undistorted 是一个 W_out×H_out 的针孔相机图像
# → 送入 VisionIPC → openpilot modeld 使用 K_undistort 作为相机内参
```

#### 修改 openpilot CameraConfig

需要在 `common/transformations/camera.py` 中定义新的相机配置：

```python
# 鱼眼去畸变后的等效针孔相机
_fisheye_undistorted = CameraConfig(W_out, H_out, f_undistort)
_fisheye_config = DeviceCameraConfig(
    fcam=_fisheye_undistorted,  # 去畸变图同时作为 fcam 和 ecam
    dcam=_NoneCameraConfig(),
    ecam=_fisheye_undistorted,
)
```

#### 方案 A 优缺点

| 优点 | 缺点 |
|------|------|
| 修改最小，复用 openpilot 完整流水线 | 两次插值（去畸变 + warp），累积误差 |
| 可利用现有标定系统（rpyCalib） | 需要维护中间图像的缓冲区 |
| 调试方便，去畸变图可直接查看 | 计算量稍大（多一步 remap） |

### 5.3 方案 B：鱼眼直接到模型输入（一步法）

```
鱼眼原图 (W×H, fisheye, HFOV=120°)
      │
      │  自定义 remap (合并 去畸变 + warp + loadyuv)
      ↓
模型输入 (1, 12, 128, 256) × 2
```

#### 核心思路

将鱼眼去畸变和 openpilot warp 变换合并为一个查找表，对每个模型输出像素直接计算其在鱼眼原图中的源坐标。

#### 查找表计算

对于模型输出图像中的每个像素 (u_model, v_model)：

```python
def compute_fisheye_to_model_map(K_fisheye, D, K_model, rpyCalib, model_size):
    """计算从模型像素到鱼眼源像素的映射表。

    数学推导:
    1. 模型像素 → 标定坐标系 3D 射线: ray = calib_from_model @ [u, v, 1]
    2. 标定坐标系 → 设备坐标系: ray_dev = device_from_calib @ ray
    3. 设备坐标系 → 视图坐标系: ray_view = view_from_device @ ray_dev
    4. 视图坐标系 3D 射线 → 鱼眼像素: 反投影鱼眼模型
    """
    W_model, H_model = model_size
    # 步骤 1-3: 与 openpilot get_warp_matrix 相同的坐标变换
    calib_from_model = ...  # calib_from_medmodel 或 calib_from_sbigmodel
    device_from_calib = rot_from_euler(rpyCalib)
    view_from_device = view_frame_from_device_frame

    # 对每个模型像素生成 3D 射线方向
    u, v = np.meshgrid(np.arange(W_model), np.arange(H_model))
    pts_model = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H, W, 3)
    rays_calib = pts_model @ calib_from_model.T               # (H, W, 3)
    rays_device = rays_calib @ device_from_calib.T
    rays_view = rays_device @ view_from_device.T

    # 步骤 4: 3D 射线 → 鱼眼像素（使用 OpenCV fisheye 反投影）
    rays_normalized = rays_view[..., :2] / rays_view[..., 2:3]  # (H, W, 2)
    rays_flat = rays_normalized.reshape(-1, 1, 2).astype(np.float64)
    fisheye_pts = cv2.fisheye.distortPoints(rays_flat, K_fisheye, D)
    map_xy = fisheye_pts.reshape(H_model, W_model, 2)

    return map_xy[..., 0].astype(np.float32), map_xy[..., 1].astype(np.float32)
```

> 注意：上面的代码是概念性的伪代码，实际实现需要注意 OpenCV fisheye 的 distortPoints API 的精确用法。

#### 方案 B 优缺点

| 优点 | 缺点 |
|------|------|
| 只有一次插值，图像质量最优 | 需要自行处理 NV12 → YUV 棋盘拆分 |
| 无中间图像缓冲，内存更低 | 跳过了 openpilot 的 OpenCL warp 内核 |
| 可直接优化为 OpenCL/CUDA | 标定更新 (rpyCalib) 时需重算查找表 |
| 计算量更少 | 调试更复杂（无法查看中间图像） |

### 5.4 方案选择建议

| 场景 | 推荐方案 |
|------|---------|
| 快速验证 / 原型开发 | **方案 A**（最少的代码改动） |
| 嵌入式部署 / 追求最优质量 | **方案 B**（一次插值，效率高） |
| 需要复用 openpilot calibrationd | **方案 A**（完整利用标定系统） |

---

## 6. 详细分辨率评估

### 6.1 1920×1080（最常见）

```
f_fish = 917 px/rad

MEDMODEL (f=910, ±15.7°):
  中心: R = 917/910 = 1.01  → 1:1 映射，不上不下
  边缘: R = 917 × cos²(15.7°) / 910 = 0.93 → 7% 上采样

SBIGMODEL (f=455, ±29.4°):
  中心: R = 917/455 = 2.02  → 2× 降采样
  边缘: R = 917 × cos²(29.4°) / 455 = 1.53 → 充足
```

**结论：勉强可行。** MEDMODEL 边缘有约 7% 的分辨率不足，车道线检测在远距离可能略有精度下降，但近距离（LDW 最关心的区域）不受影响。SBIGMODEL 完全充足。

**适用场景**：成本敏感的原型验证。如果 LDW 精度要求不高（预警距离 < 50m），1080p 可以接受。

### 6.2 2560×1440（推荐）

```
f_fish = 1222 px/rad

MEDMODEL (f=910, ±15.7°):
  中心: R = 1222/910 = 1.34 → 充足
  边缘: R = 1222 × cos²(15.7°) / 910 = 1.24 → 充足

SBIGMODEL (f=455, ±29.4°):
  中心: R = 1222/455 = 2.69 → 优秀
  边缘: R = 1222 × cos²(29.4°) / 455 = 2.04 → 优秀
```

**结论：良好，推荐选择。** 全域降采样，MEDMODEL 最差处仍有 1.24× 余量，接近 openpilot 原装 Wide Camera 的质量水平（Wide Camera 送入 SBIGMODEL 时缩放比为 567/455 = 1.25）。

### 6.3 3840×2160（最佳）

```
f_fish = 1833 px/rad

MEDMODEL (f=910, ±15.7°):
  中心: R = 1833/910 = 2.01 → 优秀
  边缘: R = 1833 × cos²(15.7°) / 910 = 1.87 → 优秀

SBIGMODEL (f=455, ±29.4°):
  中心: R = 1833/455 = 4.03 → 优秀（信息富余）
  边缘: R = 1833 × cos²(29.4°) / 455 = 3.05 → 优秀
```

**结论：完全可行，信息充裕。** MEDMODEL 的质量接近 openpilot 原装 Road Camera（fcam 2648/910 = 2.91），是最优选择。但 4K 分辨率对处理能力要求较高。

### 6.4 与 openpilot 原装相机的对比

| 配置 | MEDMODEL 中心 R | MEDMODEL 边缘 R | SBIGMODEL 中心 R | SBIGMODEL 边缘 R |
|------|----------------|----------------|-----------------|-----------------|
| **openpilot Road Camera (fcam)** | 2.91 | 2.68 | — | — |
| **openpilot Wide Camera (ecam)** | — | — | 1.25 | 0.95 |
| **鱼眼 1080p (120° HFOV)** | 1.01 | 0.93 | 2.02 | 1.53 |
| **鱼眼 1440p (120° HFOV)** | 1.34 | 1.24 | 2.69 | 2.04 |
| **鱼眼 4K (120° HFOV)** | 2.01 | 1.87 | 4.03 | 3.05 |

注意 openpilot 原装 Wide Camera 送入 SBIGMODEL 时，边缘 R=0.95 也是略有上采样的——openpilot 自己也接受了这个妥协。

---

## 7. 单相机双目模式的特殊考虑

### 7.1 什么是单相机双目模式

openpilot 模型期望两路输入（`img` + `big_img`），物理上来自两个相机（Road + Wide）。使用单颗鱼眼相机时，同一图像需要同时提供两路输入：

- `img`（MEDMODEL，fl=910，~31° HFOV）：取鱼眼中心区域的高分辨率部分
- `big_img`（SBIGMODEL，fl=455，~59° HFOV）：取鱼眼更广的区域

### 7.2 120° HFOV 足够覆盖两路输入

120° HFOV >> 59°（SBIGMODEL 需要），FOV 覆盖完全没有问题。鱼眼相机的优势在于广视角天然包含了模型需要的全部视野范围。

### 7.3 cy 偏移

MEDMODEL 的 cy = 47.6（非常偏上，主要关注地平线以下的道路区域）。物理相机需要：
- 安装时大致水平朝向前方
- 通过 openpilot 的 `liveCalibration` 系统自动估算 pitch 偏差
- warp 变换会将 pitch 偏差旋转掉

鱼眼相机安装时无需特别注意角度，标定系统会处理。但如果 pitch 偏差过大（>10°），warp 可能采样到图像边界外，需确认安装角度在合理范围内。

---

## 8. 额外需要注意的问题

### 8.1 鱼眼标定质量

鱼眼去畸变的质量完全取决于标定精度。建议：
- 使用 **ChArUco 标定板**（比普通棋盘格更鲁棒）
- 拍摄 40+ 张不同角度的标定图（覆盖画面边缘）
- 使用 `cv2.fisheye.calibrate()` 获取 K 和 D (k1-k4)
- 验证重投影误差 < 1.0 像素

### 8.2 色彩空间

openpilot modeld 期望 NV12（YUV420）格式输入。如果鱼眼相机输出 RGB/BGR：
- 方案 A：去畸变后转 YUV，送入 VisionIPC
- 方案 B：在自定义 remap 中直接处理原始格式

如果鱼眼相机直接输出 YUV（如 USB UVC 相机），可以在 YUV 域做 remap，避免色彩空间转换开销。

### 8.3 帧率

openpilot modeld 以 20 Hz 运行。鱼眼相机帧率需 ≥ 20 fps。

对于 remap 计算开销：
- 1080p cv2.remap：~3ms（CPU，单核），不影响 20 Hz
- 4K cv2.remap：~12ms（CPU，单核），仍可满足 20 Hz
- 如使用 OpenCL/CUDA remap：< 1ms

### 8.4 模型域适应

openpilot 的 driving vision 模型是在针孔相机图像上训练的。去畸变后的图像在视觉特征上与针孔图像高度一致（尤其在中心区域），但可能存在细微差异：

1. **中心区域**（MEDMODEL 关注的 ~31°）：去畸变后与针孔图几乎无差别，模型应能直接工作。
2. **边缘区域**（SBIGMODEL 的 ~59°）：去畸变插值可能引入轻微模糊，但 SBIGMODEL 本身对分辨率要求较低（fl=455），影响有限。
3. **如果去畸变质量不佳**（标定精度差或使用简单模型），残余畸变可能影响车道线检测精度。

**建议**：先直接使用 openpilot 预训练模型进行测试。如果效果不理想，可以使用 `tools/dashcam/train/` 中的微调工具，用鱼眼去畸变后的数据微调模型。

---

## 9. 实施路线图

### 阶段 1：标定与验证（1-2 天）

1. 对鱼眼相机进行 OpenCV fisheye 标定
2. 计算 remap 查找表
3. 用测试图像验证去畸变效果（直线是否变直、棋盘格是否规则）

### 阶段 2：接入 openpilot 流水线（方案 A）（2-3 天）

1. 在 `camera.py` 中添加鱼眼去畸变后的 CameraConfig
2. 在相机数据采集端（camerad 或自定义脚本）添加 cv2.remap 步骤
3. 将去畸变图像送入 VisionIPC
4. 修改 modeld 使用正确的相机内参
5. 端到端测试：检查模型输出（车道线、前车）是否合理

### 阶段 3：LDW/FCW 逻辑验证（1-2 天）

1. 在实际行车场景（或 Carla 仿真）中运行
2. 验证车道线检测精度
3. 验证前车检测距离和准确度
4. 根据结果决定是否需要微调模型

### 阶段 4（可选）：优化到一步法（方案 B）

如果方案 A 满足需求但需要优化性能，迁移到方案 B。

---

## 10. 结论

### 10.1 可行性判断

**使用 120° HFOV 鱼眼相机实现 openpilot 基于 driving vision 的 LDW/FCW 是可行的**，但需要添加鱼眼去畸变预处理步骤。

### 10.2 关键条件

| 条件 | 要求 | 说明 |
|------|------|------|
| 鱼眼分辨率 | **≥ 2560×1440（推荐）**<br>≥ 1920×1080（最低） | 120° HFOV 时，1440p 才能保证 MEDMODEL 全域无上采样 |
| 鱼眼标定 | 重投影误差 < 1.0 px | 去畸变质量直接影响感知精度 |
| 去畸变处理 | 必需（非可选） | openpilot 流水线假设针孔模型 |
| 帧率 | ≥ 20 fps | modeld 运行频率 |

### 10.3 各分辨率的最终建议

| 鱼眼分辨率 | 可行性 | LDW 效果预期 | FCW 效果预期 | 建议 |
|-----------|--------|------------|------------|------|
| 1280×720 | ❌ 不可行 | 全域模糊 | 远距离差 | 不建议 |
| 1920×1080 | ⚠️ 勉强 | 近距离可用，远距离略差 | 中距离可用 | 仅用于快速验证 |
| **2560×1440** | **✅ 推荐** | **良好** | **良好** | **性价比最优** |
| 3840×2160 | ✅✅ 最佳 | 优秀 | 优秀 | 追求最佳效果时选用 |

### 10.4 与双相机方案的对比

openpilot 原装使用双相机（窄角 Road + 广角 Wide），单颗 120° 鱼眼的方案对比：

| 维度 | openpilot 双相机 | 单颗 120° 鱼眼 (1440p) |
|------|-----------------|----------------------|
| MEDMODEL 质量 | 优秀 (2.91×) | 良好 (1.34×) |
| SBIGMODEL 质量 | 可用 (ecam 1.25× 中心) | 优秀 (2.69× 中心) |
| 硬件成本 | 高（两颗相机 + 同步） | 低（一颗相机） |
| 安装复杂度 | 高 | 低 |
| 额外软件处理 | 无 | 需要鱼眼去畸变 |

单颗鱼眼在 MEDMODEL 上不如双相机的窄角 Road Camera（因为120° 的像素被分散到更大的视场），但 SBIGMODEL 反而更优（因为分辨率高于 openpilot 原装 Wide Camera 给 SBIGMODEL 的有效分辨率）。对于 LDW/FCW 应用场景，这是一个合理的折中。

---

## 附录 A：公式推导

### A.1 等距鱼眼角分辨率

等距投影模型：r = f_fish × θ

其中 r 为像素到主点的距离，θ 为入射角（弧度），f_fish 为鱼眼焦距（px/rad）。

对于 HFOV = 2α_max：
```
f_fish = r_max / α_max = (W/2) / α_max
```

120° HFOV → α_max = 60° = π/3：
```
f_fish = (W/2) / (π/3) = 3W / (2π) ≈ 0.4775 × W
```

角分辨率（每像素覆盖的角度）：
```
dθ/dr = 1/f_fish  (常数，与 θ 无关)
```

### A.2 针孔相机角分辨率

针孔投影模型：r = f_pin × tan(θ)

角分辨率在角度 θ 处：
```
dr/dθ = f_pin / cos²(θ)
→ dθ/dr = cos²(θ) / f_pin  (随 θ 增大而减小)
```

### A.3 有效分辨率比值

模型虚拟相机在角度 θ 处每个像素对应的角度：dθ_model = cos²(θ) / f_model

鱼眼源在同一角度处每个像素对应的角度：dθ_fish = 1 / f_fish

每个模型像素对应的源像素数 = dθ_model / dθ_fish：

```
R(θ) = f_fish × cos²(θ) / f_model
```

R > 1 表示源像素充足（降采样），R < 1 表示不足（上采样，引入模糊）。

---

## 参考文件

| 文件路径 | 内容 |
|---------|------|
| `docs/image_processing_pipeline.md` | openpilot 完整图像处理流水线文档 |
| `docs/camera_selection_guide.md` | 物理相机选型指南 |
| `docs/dashcam_wide_road_only_analysis.md` | wide-road-only 模式分析 |
| `docs/tusimple-fisheye-deployment.md` | TuSimple 鱼眼部署方案（remap 实现参考） |
| `common/transformations/model.py` | MEDMODEL/SBIGMODEL 虚拟相机内参 |
| `common/transformations/camera.py` | 物理相机内参定义 |
| `selfdrive/modeld/modeld.py` | modeld 主循环（warp 矩阵计算） |
