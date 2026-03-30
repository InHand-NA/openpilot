# TuSimple 鱼眼相机部署方案

## 1. 问题背景

训练数据在 Carla 中使用**针孔 (pinhole)** 相机模型渲染（1920×1080, HFOV=120°），经 ROI 裁剪（70° HFOV）后缩放到 1280×720 作为 TuSimple 标注图像，模型输入为 640×360。

部署时使用的是一颗**鱼眼 (fisheye)** 相机（1920×1080, HFOV=120°）。鱼眼与针孔的投影模型不同，同一个 3D 点在两种相机中的成像位置不同，必须做重映射 (remap) 才能消除域差距。

### 1.1 投影模型差异

```
针孔 (Carla 训练):  r = f × tan(θ)
鱼眼 (实际部署):    r = f' × d(θ)

常见鱼眼模型:
  等距 (equidistant):    d(θ) = θ
  等立体角 (equisolid):  d(θ) = 2·sin(θ/2)
  OpenCV fisheye:        d(θ) = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
```

在图像边缘（θ=60°）投影偏差极大：

| 模型 | r/f 值 (θ=60°) | 相对针孔偏差 |
|------|----------------|------------|
| 针孔 | tan(60°) = 1.732 | 基准 |
| 等距 | π/3 = 1.047 | -39.5% |
| 等立体角 | 2·sin(30°) = 1.000 | -42.3% |

→ 不做重映射的话，边缘区域的车道线位置偏差达数百像素。

### 1.2 为什么裁到 70° 有利

70° 中心区域偏差大幅缩小（θ=35° 边缘）：

| 模型 | r/f 值 (θ=35°) | 相对针孔偏差 |
|------|----------------|------------|
| 针孔 | tan(35°) = 0.700 | 基准 |
| 等距 | 35°×π/180 = 0.611 | -12.7% |
| 等立体角 | 2·sin(17.5°) = 0.601 | -14.1% |

偏差从 ~40% 降到 ~13%，remap 插值质量更高，对标定精度要求也更低。


## 2. 有效分辨率分析

**核心问题**：鱼眼 1920×1080 原图中 70° 中心区域的像素是否足够支撑 640×360 模型输入？

### 2.1 角分辨率对比方法

> **简化假设**：以下分析以理想等距 (equidistant) 模型为基础，假设鱼眼角分辨率在整个 FOV 内均匀分布。实际部署使用的 OpenCV fisheye 模块基于 Kannala-Brandt 模型（含 k1–k4 高阶畸变系数），其角分辨率分布与等距模型接近但存在偏差——高阶系数会使边缘分辨率略高于或低于等距理论值。具体部署时应使用实际标定参数验证边缘像素对应关系。等距模型的分析结果可作为合理的近似估计。

对于目标针孔图像中角度 θ 处的一个像素，其对应的角张量为：

```
Δθ_target = cos²θ / f_target      (针孔在 θ 处每像素覆盖的角度)
```

对于等距鱼眼源图中同一角度的一个像素：

```
Δθ_fish = 1 / f_fish              (等距鱼眼: 角分辨率均匀)
```

**分辨率比值 = (f_fish × cos²θ) / f_target**：>1 表示鱼眼源像素充足（下采样），<1 表示不足（上采样）。

推导：目标针孔中一个像素覆盖的角度为 Δθ_target = cos²θ / f_target，鱼眼源中一个像素覆盖 Δθ_fish = 1 / f_fish。每个目标像素对应的源像素数 = Δθ_target / Δθ_fish = (f_fish × cos²θ) / f_target。

### 2.2 两种目标分辨率对比

鱼眼参数（等距模型）：f_fish = 960 / (π/3) ≈ 916.7 px/rad

**方案 A：鱼眼 → 640×360（直接到模型输入）**

f_target = 320 / tan(35°) ≈ 457.0 px

| θ (离轴角) | 针孔 px/rad | 鱼眼 px/rad | 比值 | 状态 |
|-----------|------------|------------|------|------|
| 0° (中心) | 457.0 | 916.7 | **2.01×** | 下采样 |
| 10° | 471.2 | 916.7 | 1.95× | 下采样 |
| 20° | 517.5 | 916.7 | 1.77× | 下采样 |
| 30° | 609.3 | 916.7 | 1.50× | 下采样 |
| 35° (边缘) | 681.1 | 916.7 | **1.35×** | 下采样 |

→ **全域下采样，最差处仍有 1.35× 余量。有效分辨率完全超过 640×360。**

**方案 B：鱼眼 → 1280×720（训练标注分辨率）**

f_target = 640 / tan(35°) ≈ 914.0 px

| θ (离轴角) | 针孔 px/rad | 鱼眼 px/rad | 比值 | 状态 |
|-----------|------------|------------|------|------|
| 0° (中心) | 914.0 | 916.7 | 1.00× | 刚好 |
| 10° | 942.4 | 916.7 | **0.97×** | 上采样 |
| 20° | 1035.1 | 916.7 | **0.89×** | 上采样 |
| 30° | 1218.7 | 916.7 | **0.75×** | 上采样 |
| 35° (边缘) | 1362.1 | 916.7 | **0.67×** | 上采样 |

→ 1280×720 目标超过了鱼眼源的有效分辨率，边缘需要上采样 1.5×，引入模糊。

### 2.3 结论

| 目标分辨率 | 中心比值 | 边缘比值 | 是否可行 |
|-----------|---------|---------|---------|
| 640×360 | 2.01× | 1.35× | 全域充足，推荐 |
| 1280×720 | 1.00× | 0.67× | 边缘不足，不推荐 |

**部署时应直接从鱼眼重映射到 640×360，跳过 1280×720 中间步骤。** 这不仅避免了分辨率不足的问题，还减少了一次插值操作。


## 3. 部署处理流程

### 3.1 总体流程

```
鱼眼原图 (1920×1080, fisheye)
         │
         │  单步 cv2.remap (预计算查找表)
         ↓
虚拟针孔图 (640×360, HFOV=70°, f=457.0)
         │
         │  送入 TuSimple 模型
         ↓
      车道线检测结果
```

训练时的两步（裁剪 1280×720 → 缩放 640×360）在部署时合并为一步 remap。

### 3.2 具体实现

#### Step 1: 鱼眼相机标定（离线，一次性）

使用棋盘格或 ChArUco 标定板，获取鱼眼内参和畸变系数：

```python
import cv2
import numpy as np

# 标定结果
K_fisheye = np.array([...])  # 3×3 鱼眼内参
D = np.array([k1, k2, k3, k4])  # OpenCV fisheye 畸变系数
```

**关于鱼眼主点偏移**：标定得到的 K_fisheye 中主点 (cx, cy) 可能不等于图像中心 (960, 540)，偏移数十像素属于正常现象（镜头安装偏心、传感器对位误差）。**无需手动修正**——`cv2.fisheye.initUndistortRectifyMap` 会根据 K_fisheye 中的实际主点正确计算 remap 查找表。只需确保使用标定原始值，不要人为"归中"。

#### Step 2: 计算目标虚拟针孔内参（关键：cy 偏移）

训练时 crop 框不是以光轴为中心裁剪的，而是**刻意向下平移**，让地平线出现在图像顶部 30% 处，把 70% 的像素预算分配给路面：

```
原图 1920×1080:
  y=322 ─── 如果以光轴为中心裁剪的顶边
                ↕ 下移 48px
  y=370 ─── 实际裁剪的顶边
                │  天空 30%
  y=501 ─── 地平线 (pitch=4°)
                │
  y=540 ─── 光轴 (主点)          ← 不在裁剪区域中心
                │  道路 70%
  y=806 ─── 实际裁剪的底边
```

这在内参矩阵中的体现是 **cy ≠ 图像高度/2**：

| | cy | 图像中心 | cy 位置 | 地平线 | 道路占比 |
|--|---|---------|---------|-------|---------|
| 如果居中裁剪 | 360.0 (1280×720) | 360.0 | 正中 50% | 41.1% | 58.9% |
| **训练实际** | **280.4** (1280×720) | 360.0 | **偏上 38.9%** | **30.1%** | **69.9%** |
| **部署 640×360** | **140.2** | 180.0 | **偏上 38.9%** | **30.1%** | **69.9%** |

在 remap 中，K_model 的 cy 偏移直接控制虚拟相机"看向哪里"：输出图像中心行 (v=180) 通过 K_model⁻¹ 反投影出的 3D 射线方向指向光轴**下方**，等效于 crop 框下移。cy=140.2 < 180 自动实现"多取路面、少取天空"。

**精确值必须从训练时的 K_crop 等比例缩放得到**：

```python
from openpilot.tools.dashcam.tusimple.config import compute_crop_params

K_crop_1280 = compute_crop_params()['K_crop']
# K_crop_1280 = [[914.3, 0, 640.0],
#                [0, 914.3, 280.4],   ← cy=280.4, 不是 360
#                [0, 0, 1]]

# 等比缩放到 640×360
K_model = K_crop_1280 / 2
K_model[2, 2] = 1.0
# K_model = [[457.1, 0, 320.0],
#            [0, 457.1, 140.2],      ← cy=140.2, 不是 180
#            [0, 0, 1]]
```

**警告**：如果错误地将 cy 设为图像中心 (180.0)，地平线会跑到 41%，道路区域从 70% 缩减到 59%，与训练数据产生域差距。

#### Step 3: 预计算 remap 查找表（离线，一次性）

```python
# R: 校正相机安装旋转偏差（尤其 roll），详见 Section 7
# 安装角度在设计范围内时用单位矩阵即可
R = np.eye(3)  # 如需 roll/pitch/yaw 补偿，见 Section 7.2

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K=K_fisheye,       # 鱼眼标定内参
    D=D,               # 鱼眼畸变系数
    R=R,               # 旋转修正
    P=K_model,         # 目标虚拟针孔内参
    size=(MODEL_W, MODEL_H),  # (640, 360)
    m1type=cv2.CV_16SC2,       # 定点数格式，remap 更快
)
# map1, map2 缓存到文件，部署时直接加载
```

#### Step 4: 运行时每帧处理

```python
# frame_fisheye: 鱼眼原图 1920×1080
frame_model = cv2.remap(frame_fisheye, map1, map2, cv2.INTER_LINEAR)
# frame_model: 640×360 虚拟针孔图像，HFOV=70°
# → 直接送入 TuSimple 模型推理
```

单次 `cv2.remap` 将「鱼眼去畸变 + 裁剪 + 缩放」合并为一步双线性插值。

### 3.3 一步法 vs 两步法

**一步法（方案 A，推荐）**：鱼眼 → 640×360 虚拟针孔（单次 remap）
**两步法（方案 B，备选）**：鱼眼 → 1920×1080 去畸变针孔 → 裁剪缩放 640×360

方案 A 相比方案 B 的优势：

| 比较维度 | 方案 A（一步法） | 方案 B（两步法） |
|---------|----------------|----------------|
| 插值次数 | 1 次 | 2 次（remap + resize），边缘损失约 0.5–1 px |
| 计算量 | remap 640×360 = 23 万像素 | remap 1920×1080 = 207 万像素 + crop + resize |
| 内存 | 无中间缓冲区 | 需分配 1920×1080 中间图 (~6 MB) |
| 120° 边缘质量 | 不涉及（直接跳到 70°） | 边缘 θ=60° 处角分辨率仅中心的 25%，质量差（但会被裁掉） |

**方案 A 是首选**。但方案 B 在特定场景下有独特价值——当需要完整 120° 去畸变图像用于可视化、多算法复用或调试对比时，两步法提供了一个与训练流程完全对称的中间产物。详见 Section 3.5。

### 3.4 remap 质量验证

remap 查找表计算完成后，必须在实际部署前进行定量验证，确保几何变换的正确性。

#### 3.4.1 标定板重投影验证（推荐）

使用棋盘格标定板作为已知几何参照：

```python
import cv2
import numpy as np

def validate_remap(frame_fisheye, map1, map2, K_model, pattern_size=(9, 6), square_size=0.025):
    """使用棋盘格验证 remap 质量。

    Args:
        frame_fisheye: 鱼眼原图（含标定板）
        map1, map2: 预计算的 remap 查找表
        K_model: 目标虚拟针孔内参 (640×360)
        pattern_size: 棋盘格内角点数 (cols, rows)
        square_size: 棋盘格方格物理尺寸 (米)
    Returns:
        mean_error: 平均重投影误差 (像素)
    """
    # 1) remap 后检测角点
    frame_model = cv2.remap(frame_fisheye, map1, map2, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame_model, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ret:
        print("ERROR: 角点检测失败，请确保标定板在 70° FOV 内且清晰可见")
        return None

    # 亚像素精化
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # 2) 构造 3D 物理坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # 3) solvePnP 求解位姿，再反投影计算误差
    dist_coeffs = np.zeros(4)  # remap 后的虚拟针孔图无畸变
    ret, rvec, tvec = cv2.solvePnP(objp, corners, K_model, dist_coeffs)
    reproj_pts, _ = cv2.projectPoints(objp, rvec, tvec, K_model, dist_coeffs)
    errors = np.linalg.norm(corners.reshape(-1, 2) - reproj_pts.reshape(-1, 2), axis=1)

    mean_error = errors.mean()
    max_error = errors.max()
    print(f"重投影误差: mean={mean_error:.3f}px, max={max_error:.3f}px")
    return mean_error
```

**验收标准**：
- 平均重投影误差 < 1.0 px（640×360 分辨率上）
- 最大重投影误差 < 2.0 px
- 超过此阈值说明标定参数或 remap 查找表有误，需排查

#### 3.4.2 直线性定性验证

remap 后的虚拟针孔图像中，3D 空间中的直线应映射为图像中的直线。验证方法：

1. 将相机对准含有明显直线的场景（建筑物边缘、道路标线、标定板边缘）
2. 在 remap 后的图像中沿这些直线取若干采样点
3. 拟合直线，计算采样点到拟合线的最大偏差
4. 偏差应 < 1 px；如边缘区域出现弯曲，说明鱼眼畸变系数不准确

#### 3.4.3 验证时机

| 时机 | 验证内容 | 方法 |
|------|---------|------|
| 首次标定完成 | K_fisheye、D 的正确性 | 标定板重投影 (3.4.1) |
| remap 查找表生成后 | map1/map2 与 K_model 的一致性 | 标定板重投影 + 直线性 (3.4.1 + 3.4.2) |
| 更换镜头或相机模组 | 新标定参数的有效性 | 完整验证流程 |
| 环境温度极端变化时 | 热漂移影响评估（见 Section 8.6） | 标定板重投影，对比常温基准 |

### 3.5 方案 B：两步法部署（备选）

当算力不是瓶颈，且需要完整 120° 去畸变图像时，可使用两步法作为备选方案。该方案的中间产物是一张与 Carla 渲染完全同参数的 1920×1080 针孔图像，便于可视化调试和多算法复用。

#### 3.5.1 总体流程

```
鱼眼原图 (1920×1080, fisheye, HFOV=120°)
         │
         │  Step 1: cv2.remap (预计算查找表)
         ↓
虚拟针孔图 (1920×1080, pinhole, HFOV=120°, f=554.3)  ← 与 Carla 渲染同参数
         │
         │  Step 2: crop (572, 370, 776, 436)
         ↓
裁剪区域 (776×436, HFOV=70°)
         │
         │  Step 3: cv2.resize
         ↓
模型输入 (640×360, HFOV=70°, 等效 f=457.1)
         │
         │  送入 TuSimple 模型
         ↓
      车道线检测结果
```

训练时的路径为 `Carla 针孔 1920×1080 → crop → resize 640×360`，方案 B 的部署路径与此**完全对称**——区别仅在于输入从 Carla 渲染变为鱼眼 remap。

#### 3.5.2 中间图分辨率验证

中间产物为 1920×1080、HFOV=120° 的虚拟针孔图。需验证其 70° 中心区域的分辨率是否充足。

中间针孔参数：f_pinhole = 960 / tan(60°) ≈ 554.3 px/rad（= Carla 的 `MONO_FOCAL`）

| θ (离轴角) | 中间针孔 px/rad | 鱼眼 px/rad | 比值 | 状态 |
|-----------|----------------|------------|------|------|
| 0° (中心) | 554.3 | 916.7 | **1.65×** | 下采样 |
| 10° | 571.5 | 916.7 | 1.60× | 下采样 |
| 20° | 627.7 | 916.7 | 1.46× | 下采样 |
| 30° | 739.1 | 916.7 | 1.24× | 下采样 |
| 35° (70° 边缘) | 826.1 | 916.7 | **1.11×** | 下采样 |
| 60° (120° 边缘) | 2217.2 | 916.7 | **0.41×** | 严重上采样 |

**关键结论**：

- **70° 中心区域全域下采样**（最差处 1.11×），remap 质量有保证
- 120° 边缘严重上采样（0.41×），但这部分会被 crop 裁掉，**不影响模型输入质量**
- 后续 crop 776×436 → resize 640×360 是下采样（0.82×），不引入额外模糊

综合两步的等效分辨率比值（70° 边缘）：1.11 × (776/640) = **1.35×**，与方案 A 的 1.35× 一致——最终模型输入质量相同，差异仅来自双重插值的微小损失。

#### 3.5.3 具体实现

##### Step 1: 计算中间虚拟针孔内参

中间图需与 Carla 渲染的相机参数完全一致：

```python
import cv2
import numpy as np
from openpilot.tools.dashcam.tusimple.config import K_MONO, compute_crop_params

# K_MONO 即 Carla 渲染相机的内参，直接复用
# K_MONO = [[554.3, 0, 960.0],
#           [0, 554.3, 540.0],    ← cy = 图像中心 (主点居中)
#           [0, 0, 1]]
K_pinhole = K_MONO.copy()
```

注意与方案 A 的 K_model 的区别：

| | K_model（方案 A 目标） | K_pinhole（方案 B 中间图） |
|--|----------------------|------------------------|
| 分辨率 | 640×360 | 1920×1080 |
| 焦距 f | 457.1 | 554.3 |
| cx | 320.0 | 960.0 |
| cy | **140.2**（含 crop 偏移） | **540.0**（图像中心） |
| crop 偏移 | 编码在 cy 中 | 由后续 crop_rect 处理 |

方案 A 将 crop 偏移"折叠"进了 K_model 的 cy；方案 B 则保持中间图主点居中，偏移由 crop_rect 实现。两者数学等效。

##### Step 2: 预计算 remap 查找表（离线，一次性）

```python
# R: 旋转修正矩阵，含义与方案 A 相同（见 Section 7）
R = np.eye(3)

# 鱼眼 → 1920×1080 虚拟针孔 remap 表
map1_full, map2_full = cv2.fisheye.initUndistortRectifyMap(
    K=K_fisheye,          # 鱼眼标定内参
    D=D,                  # 鱼眼畸变系数
    R=R,                  # 旋转修正
    P=K_pinhole,          # 目标: 1920×1080 针孔 (= Carla 相机)
    size=(1920, 1080),    # 中间图分辨率
    m1type=cv2.CV_16SC2,  # 定点格式
)
# 缓存 map1_full, map2_full 到文件
```

##### Step 3: 获取 crop 参数

```python
params = compute_crop_params()  # pitch=4°, crop_hfov=70°, horizon_ratio=0.3
crop_rect = params['crop_rect']  # (572, 370, 776, 436)
x, y, w, h = crop_rect
```

crop_rect 与 Carla 数据采集时使用的完全相同。这是方案 B 的核心优势——**裁剪参数直接复用训练配置，无需推导 K_model**。

##### Step 4: 运行时每帧处理

```python
def process_frame_plan_b(frame_fisheye, map1_full, map2_full, crop_rect):
    """方案 B: 两步法处理鱼眼帧。

    Args:
        frame_fisheye: 1920×1080 鱼眼原图 (BGR, uint8)
        map1_full, map2_full: 预计算的 remap 查找表
        crop_rect: (x, y, w, h) 裁剪区域
    Returns:
        frame_model: 640×360 模型输入
        frame_pinhole: 1920×1080 中间去畸变图（可选，用于可视化）
    """
    x, y, w, h = crop_rect

    # Step 1: 鱼眼 → 1920×1080 虚拟针孔
    frame_pinhole = cv2.remap(frame_fisheye, map1_full, map2_full, cv2.INTER_LINEAR)

    # Step 2: 裁剪 70° 中心区域
    frame_crop = frame_pinhole[y:y+h, x:x+w]  # 776×436

    # Step 3: 缩放到模型输入分辨率
    frame_model = cv2.resize(frame_crop, (640, 360), interpolation=cv2.INTER_LINEAR)

    return frame_model, frame_pinhole
```

如果不需要中间图，可以优化为只 remap 裁剪区域（见 Section 3.5.5 优化建议）。

#### 3.5.4 方案对比

| 维度 | 方案 A（一步法） | 方案 B（两步法） |
|------|----------------|----------------|
| **插值次数** | 1 次 | 2 次 |
| **模型输入质量** | 基准 | 双重插值损失约 0.5–1 px（70° 边缘） |
| **remap 像素数** | 23 万 (640×360) | 207 万 (1920×1080) |
| **典型耗时 (x86 i7)** | ~0.5 ms | ~3.5 ms (remap ~2.5 ms + crop+resize ~1 ms) |
| **典型耗时 (ARM A76)** | ~1.5 ms | ~10 ms |
| **内存开销** | 查找表 ~1.8 MB | 查找表 ~16 MB + 中间图 ~6 MB |
| **中间产物** | 无 | 1920×1080 去畸变针孔图 |
| **cy 处理** | 编码在 K_model 中 | 由 crop_rect 处理（更直观） |
| **与训练流程对称性** | 等效但路径不同 | **完全对称** |
| **调试便利性** | 无法直接与 Carla 图像对比 | 中间图可直接与 Carla 渲染图叠加比较 |

#### 3.5.5 适用场景与优化建议

**推荐使用方案 B 的场景**：

1. **开发调试阶段**：需要将 remap 中间图与 Carla 渲染图叠加对比，验证标定和 remap 的正确性
2. **多算法复用**：除车道线检测外，还有其他算法（目标检测、自由空间等）需要使用同一帧去畸变图像，且各自有不同的 ROI 裁剪需求
3. **可视化需求**：需要在完整 120° 视野的去畸变图上绘制检测结果，供人工审查
4. **算力充裕的平台**：如 x86 开发机、带 GPU 的嵌入式平台，额外 3 ms 开销可忽略

**优化建议**：

如果不需要完整 1920×1080 中间图，可以只对裁剪区域计算 remap，减少无用像素的计算：

```python
# 优化: 只预计算 crop 区域对应的 remap 表
# 原理: map1_full 中 crop 区域外的像素永远不会被使用
map1_crop = map1_full[y:y+h, x:x+w].copy()
map2_crop = map2_full[y:y+h, x:x+w].copy()

# 运行时: remap 只输出 776×436
frame_crop = cv2.remap(frame_fisheye, map1_crop, map2_crop, cv2.INTER_LINEAR)
frame_model = cv2.resize(frame_crop, (640, 360), interpolation=cv2.INTER_LINEAR)
```

此优化使 remap 像素数从 207 万降到 34 万（776×436），耗时接近方案 A，但仍保留两次插值。适合"不需要中间图但希望使用 crop_rect 参数"的场景。

> **注意**：上述 `map1_crop` 裁剪方式成立的前提是 `initUndistortRectifyMap` 生成的 map 中每个像素的值是源图坐标（指向鱼眼原图），与目标图中的位置无关。这对 `cv2.remap` 是成立的——map 的行列索引对应输出像素位置，map 值对应输入像素位置。裁剪 map 等价于只生成目标图的一个子区域。


## 4. 训练与部署的一致性

### 4.1 等效关系

```
训练路径:
  Carla 针孔 1920×1080 (120°)
  → crop (572, 370, 776, 436) → resize 1280×720 → resize 640×360
  等效: f=914.3 的 1280×720 针孔 → 缩放到 f=457.1 的 640×360

部署路径:
  鱼眼 1920×1080 (120°)
  → remap to 640×360 (K_model, f=457.1)
  等效: 直接从鱼眼提取 f=457.1 的 640×360 针孔视图
```

两条路径产生**相同内参**的 640×360 针孔图像，模型输入一致。

### 4.2 关键参数对齐清单

| 参数 | 训练值 | 部署值 | 来源 | 严格匹配 |
|------|-------|-------|------|---------|
| 模型输入分辨率 | 640×360 | 640×360 | 模型架构 | 必须 |
| 有效 HFOV | 70° | 70° | `CROP_HFOV` | 必须 |
| 等效焦距 f | 457.1 px | 457.1 px | `K_crop[0,0] / 2` | 必须 |
| cx | 320.0 | 320.0 | 图像水平中心 | 必须 |
| **cy** | **140.2** | **140.2** | **`K_crop[1,2] / 2`** | **必须** |
| 地平线位置 | ~30% (pitch=4°) | 随实际安装角浮动 | cy 偏移 + pitch | 自然浮动 |

**cy 是最容易犯错的参数**。它不等于图像高度的一半 (180)，而是 140.2，对应训练时 crop 框的刻意下移。


## 5. 安装校准：消失点十字标记

安装时需要调整相机 pitch 尽量接近设计的 4° 标准角度。方法是在 remap 后的画面上叠加十字标记，指示 4° pitch 对应的道路消失点目标位置，安装人员调整相机直到实际道路消失点与十字对齐。

### 5.1 原理

平坦直线道路的消失点 (vanishing point) 是车道线延伸到无穷远处在图像上的汇聚点。注意：消失点**不是**光轴在图像上的投影（那是主点 (cx, cy)=(320, 140.2)）。对于下倾 pitch 角的相机，消失点出现在主点**上方**，偏移量为 f·tan(pitch)。

对于已知 pitch 和 yaw 的相机，消失点在 640×360 虚拟针孔图中的位置为：

```
vp_x = cx + f × tan(yaw)    = 320.0 + 457.1 × tan(yaw)
vp_y = cy - f × tan(pitch)  = 140.2 - 457.1 × tan(pitch)
```

设计目标 pitch=4°、yaw=0° 时：

```
vp_target = (320.0, 108.2)    ← 距图像顶部 30.1%
```

### 5.2 不同 pitch 下消失点 y 位置

| pitch | vp_y | 占图高 | 备注 |
|-------|------|-------|------|
| 0° | 140.2 | 38.9% | 水平安装，道路偏少 |
| 2° | 124.2 | 34.5% | |
| 3° | 116.2 | 32.3% | 推荐范围下限 |
| **4°** | **108.2** | **30.1%** | **设计目标** |
| 5° | 100.2 | 27.8% | 推荐范围上限 |
| 7° | 84.1 | 23.4% | 训练覆盖上限 |

每偏离 1° pitch，消失点移动约 8px（640×360 图中），肉眼可分辨。

### 5.3 叠加图形设计

在 remap 后的 640×360 画面上叠加三层标记：

```
640×360 remap 画面:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  y=84  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │ 虚线: 7° 上限
  │                                                  │
  │  y=108 ━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━ │ 十字: 4° 目标
  │                      ┃                           │
  │  y=140 ─ ─ ─ ─ ─ ─ ─┃─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │ 虚线: 0° 下限
  │                      ┃                           │
  │                                                  │
  │           (道路区域)                               │
  │                                                  │
  └──────────────────────────────────────────────────┘
       x=320
```

三层标记：

| 图层 | 内容 | 颜色 | 含义 |
|------|------|------|------|
| 目标十字 | (320, 108) 处短十字线 | 绿色实线 | pitch=4° 设计目标 |
| 推荐区间 | y=100 ~ y=116 半透明带 | 绿色半透明 | pitch 3°~5° 推荐范围 |
| 训练边界 | y=84 和 y=140 水平虚线 | 黄色虚线 | pitch 0°~7° 训练覆盖极限 |

### 5.4 实现代码

```python
import cv2
import numpy as np
from openpilot.tools.dashcam.tusimple.config import compute_crop_params

# ── 参数计算（一次性）──────────────────────────────────────
K_crop = compute_crop_params()['K_crop']
K_model = K_crop / 2
K_model[2, 2] = 1.0
f  = K_model[0, 0]   # 457.1
cx = K_model[0, 2]    # 320.0
cy = K_model[1, 2]    # 140.2

MODEL_W, MODEL_H = 640, 360

def pitch_to_vp_y(pitch_deg: float) -> int:
    """pitch 角度 → 消失点 y 坐标 (640×360)"""
    return int(round(cy - f * np.tan(np.radians(pitch_deg))))

# 预计算关键 y 坐标
VP_TARGET_Y = pitch_to_vp_y(4.0)   # 108  设计目标
VP_MIN_Y    = pitch_to_vp_y(5.0)   # 100  推荐范围顶部 (pitch 大 → y 小)
VP_MAX_Y    = pitch_to_vp_y(3.0)   # 116  推荐范围底部
TRAIN_TOP_Y = pitch_to_vp_y(7.0)   # 84   训练上限
TRAIN_BOT_Y = pitch_to_vp_y(0.0)   # 140  训练下限
VP_X        = int(round(cx))        # 320

# ── 每帧叠加 ──────────────────────────────────────────────
def draw_vanishing_point_guide(frame: np.ndarray) -> np.ndarray:
    """在 640×360 remap 图上叠加消失点校准标记。

    Args:
        frame: 640×360 BGR 图像（remap 后的虚拟针孔图）
    Returns:
        带叠加标记的图像（原图被修改）
    """
    # 1) 推荐区间半透明带 (pitch 3°~5°)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, VP_MIN_Y), (MODEL_W, VP_MAX_Y), (0, 200, 0), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # 2) 训练覆盖边界 (pitch 0° 和 7°) — 黄色虚线
    for y_bound in [TRAIN_TOP_Y, TRAIN_BOT_Y]:
        for x_start in range(0, MODEL_W, 12):
            cv2.line(frame, (x_start, y_bound), (min(x_start + 6, MODEL_W), y_bound),
                     (0, 200, 255), 1)

    # 3) 目标十字 (pitch=4°) — 绿色实线
    cross_len = 20
    cv2.line(frame, (VP_X - cross_len, VP_TARGET_Y),
             (VP_X + cross_len, VP_TARGET_Y), (0, 255, 0), 2)
    cv2.line(frame, (VP_X, VP_TARGET_Y - cross_len),
             (VP_X, VP_TARGET_Y + cross_len), (0, 255, 0), 2)

    # 4) 标注文字
    cv2.putText(frame, '4deg', (VP_X + cross_len + 4, VP_TARGET_Y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(frame, '7deg', (MODEL_W - 40, TRAIN_TOP_Y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
    cv2.putText(frame, '0deg', (MODEL_W - 40, TRAIN_BOT_Y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)

    return frame
```

### 5.5 安装校准流程

1. 启动校准程序，实时显示 remap 后的 640×360 画面 + 叠加标记
2. 将相机对准一段直线道路
3. 观察道路远方的实际消失点（道路两侧线条汇聚处）
4. 调整相机 pitch，使实际消失点落在绿色十字上（精确）或绿色半透明带内（可接受）
5. 调整相机 yaw，使实际消失点水平居中（x≈320）
6. 确认实际消失点不超出黄色虚线范围
7. 固定相机，完成安装

### 5.6 消失点不在十字上的诊断

| 实际消失点位置 | 原因 | 调整方向 |
|-------------|------|---------|
| 十字上方 | pitch 过大 (>4°) | 抬高相机尾部 / 压低相机头部 |
| 十字下方 | pitch 过小 (<4°) | 压低相机尾部 / 抬高相机头部 |
| 十字左侧 | yaw 偏右 | 相机向左旋转 |
| 十字右侧 | yaw 偏左 | 相机向右旋转 |
| 消失点模糊/不存在 | 弯道或非直线道路 | 换一段直线道路校准 |


### 5.7 自动消失点检测与量化输出

Section 5.5 的人工目视校准依赖操作人员经验，适合安装调试阶段。为支持批量部署和开机自检（Section 8.5 方案 2），需要自动化的消失点检测与定量偏差输出。

#### 5.7.1 检测算法

基于直线道路场景下车道线汇聚特性，自动估计消失点位置：

```python
import cv2
import numpy as np

def detect_vanishing_point(frame: np.ndarray,
                           roi_y_range=(60, 180),
                           min_line_length=40) -> tuple[float, float] | None:
    """自动检测 640×360 remap 图中的道路消失点。

    方法: Canny 边缘 → HoughLinesP → 筛选近似平行于车道线方向的线段
         → 两两求交点 → RANSAC 投票得到消失点。

    Args:
        frame: 640×360 BGR remap 图像
        roi_y_range: 消失点搜索 y 范围 (排除天空和近处路面)
        min_line_length: HoughLinesP 最小线段长度
    Returns:
        (vp_x, vp_y) 消失点坐标，检测失败返回 None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 只保留下半部分边缘 (车道线区域)
    mask = np.zeros_like(edges)
    mask[roi_y_range[0]:, :] = 255
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                             minLineLength=min_line_length, maxLineGap=10)
    if lines is None or len(lines) < 2:
        return None

    # 筛选: 保留倾斜角在 20°~80° 的线段 (排除近水平/近垂直噪声)
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 20 < angle < 80:
            filtered.append((x1, y1, x2, y2))

    if len(filtered) < 2:
        return None

    # 两两求交点，RANSAC 投票
    intersections = []
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            pt = _line_intersection(filtered[i], filtered[j])
            if pt is not None:
                px, py = pt
                # 只保留合理范围内的交点
                if 100 < px < 540 and roi_y_range[0] < py < roi_y_range[1]:
                    intersections.append(pt)

    if len(intersections) < 3:
        return None

    # 取中位数作为鲁棒估计
    pts = np.array(intersections)
    vp_x = np.median(pts[:, 0])
    vp_y = np.median(pts[:, 1])
    return (float(vp_x), float(vp_y))


def _line_intersection(l1, l2):
    """两条线段的交点 (齐次坐标法)。"""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (px, py)
```

#### 5.7.2 偏差量化与判定

```python
def quantify_calibration(vp_detected: tuple[float, float],
                         f: float = 457.1,
                         cx: float = 320.0,
                         cy: float = 140.2) -> dict:
    """将检测到的消失点转换为 pitch/yaw 偏差角度。

    Returns:
        字典含 pitch_deg, yaw_deg, pitch_status, yaw_status
    """
    vp_x, vp_y = vp_detected
    pitch_deg = np.degrees(np.arctan((cy - vp_y) / f))
    yaw_deg = np.degrees(np.arctan((vp_x - cx) / f))

    # 判定状态
    if 3.0 <= pitch_deg <= 5.0:
        pitch_status = "OK (推荐范围内)"
    elif 0.0 <= pitch_deg <= 7.0:
        pitch_status = "WARN (训练覆盖范围内，但偏离推荐值)"
    else:
        pitch_status = "ERROR (超出训练覆盖范围)"

    yaw_status = "OK" if abs(yaw_deg) < 3.0 else "WARN (yaw 偏差较大)"

    return {
        'pitch_deg': round(pitch_deg, 2),
        'yaw_deg': round(yaw_deg, 2),
        'pitch_status': pitch_status,
        'yaw_status': yaw_status,
    }
```

输出示例：
```
消失点检测: vp=(318.5, 109.1)
pitch=3.87°  [OK (推荐范围内)]
yaw=-0.19°   [OK]
```

#### 5.7.3 应用场景

| 场景 | 调用方式 | 处理逻辑 |
|------|---------|---------|
| 安装校准 (Section 5.5) | 实时显示 + 定量输出 | 辅助人工调整，显示偏差数值 |
| 开机自检 (Section 8.5 方案 2) | 启动后取前 N 帧直线道路的中位数 | pitch/yaw 超出训练范围则告警并记录日志 |
| 定期巡检 | 后台每 M 帧检测一次 | 持续监控外参漂移，偏差超阈值则提示重新校准 |

**局限性**：该方法要求场景中存在近似直线的车道线或道路边缘。弯道、无标线的道路上检测会失败，此时应降级为不输出判定结果，而非给出错误估计。


## 6. 像素值 Pipeline 一致性

几何变换（remap）只是训练-部署一致性的一半，像素值域的对齐同样重要。

### 6.1 色彩空间与归一化

| 环节 | 训练 (Carla) | 部署 (鱼眼相机) | 注意事项 |
|------|-------------|---------------|---------|
| 色彩空间 | Carla 输出 BGRA/BGR | 相机通常输出 BGR (OpenCV) 或 YUV | 确保送入模型前统一为 BGR |
| 数值范围 | 0–255 uint8 | 0–255 uint8 | 如训练时做了 /255.0 归一化，部署时必须相同 |
| Gamma | Carla 默认 sRGB gamma | 相机 ISP 也输出 sRGB | 通常一致，无需额外处理 |

**检查项**：确认训练代码中 `dataset.py` 对像素的预处理操作（归一化、通道顺序、数据类型），在推理代码中严格复现。

### 6.2 Carla 渲染 vs 真实相机的域差距

Carla 渲染图像与真实鱼眼相机在以下方面存在系统性差异：

- **动态范围**：Carla 无过曝/欠曝，真实相机在逆光/隧道场景下会出现
- **噪声**：Carla 图像无传感器噪声，真实相机在低光照下噪声明显
- **运动模糊**：Carla 默认无运动模糊，真实相机在急转弯时会出现
- **ISP 差异**：真实相机的白平衡、锐化、降噪等 ISP 处理会改变纹理特征

这些差异是 sim-to-real 域迁移的固有问题。当前方案通过 70° 裁剪减少了边缘畸变差异，但像素级域差距需要通过以下分阶段策略缓解：

#### 阶段 1：训练时数据增强（零成本，立即可做）

在训练 pipeline 中对 Carla 渲染图像施加随机增强，模拟真实相机特性：

| 增强类型 | 实现方式 | 参数建议 | 模拟目标 |
|---------|---------|---------|---------|
| 亮度/对比度 | `torchvision.transforms.ColorJitter` | brightness=0.3, contrast=0.3 | ISP 自动曝光差异 |
| 色调/饱和度 | `ColorJitter` | hue=0.05, saturation=0.3 | 白平衡差异 |
| 高斯噪声 | 自定义 transform | sigma=0~15 (uint8 尺度) | 低光照传感器噪声 |
| 运动模糊 | 随机方向线性核卷积 | kernel_size=3~7 | 急转弯/颠簸 |
| 随机阴影 | 半透明多边形叠加 | alpha=0.3~0.6 | 树荫/建筑遮挡 |

**注意**：增强只作用于输入图像，不改变标注标签。增强概率建议 50%（每帧独立），避免模型过拟合到增强模式。

#### 阶段 2：真实数据微调（需数据采集）

当具备真实鱼眼相机的采集条件时：

- **最小数据量**：建议至少 2000 帧标注数据（覆盖晴天/阴天/傍晚三种光照条件），用于微调最后 2~3 层
- **场景多样性**：至少覆盖 3 种道路类型（高速/城市/乡村）、2 种天气（晴/阴）、2 种时段（白天/傍晚）
- **微调策略**：冻结 backbone，仅微调输出头，学习率设为 Carla 预训练时的 1/10
- **混合训练**：建议 Carla 数据与真实数据按 1:1 混合，防止在真实小数据集上过拟合

#### 阶段 3：部署初期置信度策略

在仅有 Carla 训练模型、尚未经过真实数据微调时，部署应采用保守策略：

- **仅启用告警功能**（LDW/FCW），不参与车辆控制
- **输出附带不确定性**：利用模型 MDN 输出的 sigma 参数，sigma 过大时抑制告警
- **设定触发阈值高于最终目标**：例如 LDW 偏离阈值暂设为 0.5m（最终 0.3m），降低误报率
- **采集部署期间的推理结果**：作为阶段 2 微调的数据来源（主动学习 pipeline）

### 6.3 remap 插值方法一致性

训练时 crop + resize 使用的插值方法应与部署时 remap 的插值方法一致：

| 操作 | 训练默认 | 部署 remap | 建议 |
|------|---------|-----------|------|
| 缩小 (downsample) | `cv2.resize` 默认 INTER_LINEAR | `cv2.remap` INTER_LINEAR | 一致 ✓ |
| 替代方案 | INTER_AREA（抗锯齿更好） | remap 不支持 INTER_AREA | 如训练用 INTER_AREA 需注意差异 |

建议训练和部署统一使用 `INTER_LINEAR`，或在训练时验证 INTER_AREA 与 INTER_LINEAR 的精度差异可忽略。


## 7. R 矩阵：安装旋转校正

Section 3.2 Step 3 中的 R 矩阵用于在 remap 中补偿相机安装旋转偏差。

### 7.1 何时需要 R ≠ I

| 情况 | R 矩阵 | 说明 |
|------|--------|------|
| 安装角度在设计范围内 (pitch 3°–5°, yaw < 3°, roll ≈ 0°) | **R = I** | 模型已学习适应该范围的变化 |
| 存在明显 roll（水平倾斜 > 2°） | **R = R_roll** | roll 会导致车道线在图像中倾斜，模型未训练该变体 |
| pitch/yaw 偏差大但无法机械调整 | **R = R_pitch · R_yaw** | 通过 R 矩阵在 remap 中补偿到设计角度附近 |

### 7.2 R 矩阵计算

```python
import numpy as np

def rotation_matrix(roll_deg=0, pitch_deg=0, yaw_deg=0):
    """计算从实际安装姿态到目标姿态的旋转修正矩阵。

    参数为需要补偿的偏差角度（实际值 - 目标值）。
    例如：实际 roll=3°，目标 roll=0°，则 roll_deg=-3 以补偿回来。
    """
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx

# 示例：补偿 3° roll 和 1° yaw 偏差
R = rotation_matrix(roll_deg=-3, yaw_deg=-1)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K=K_fisheye, D=D, R=R, P=K_model,
    size=(640, 360), m1type=cv2.CV_16SC2
)
```

### 7.3 Roll 校正的重要性

在所有安装偏差中，**roll 是最需要通过 R 矩阵校正的**：
- pitch/yaw 偏差：模型训练时已覆盖一定范围（pitch 0°–7°，yaw ±3°），小偏差可容忍
- roll 偏差：训练数据中 roll 始终为 0°，**任何 roll 偏差都是域外数据**，直接导致车道线在图像中倾斜，影响检测精度

建议安装后先检查画面水平度，如 roll > 1° 则将补偿值写入 R 矩阵。


## 8. 其他注意事项

### 8.1 标定精度要求

70° 中心区域对标定误差的容忍度较高。经验值：
- 焦距误差 < 2%：车道线位置偏差 < 3px (在 640×360 上)
- 畸变系数精度：棋盘格 20+ 张多角度图像通常足够
- 建议使用 OpenCV `cv2.fisheye.calibrate()` 标准流程

### 8.2 安装角度

训练数据覆盖 pitch 0°~7°、yaw -3°~+3°。部署时：
- 安装 pitch 应在此范围内（推荐 3°~5°）
- 无需精确标定 pitch — 模型已学习适应地平线位置变化
- yaw 偏差 < 3° 不需要修正

### 8.3 不同鱼眼模型的适配

OpenCV `cv2.fisheye` 模块使用 Kannala-Brandt 模型（等距变种）。如果相机使用其他鱼眼模型：
- 等距/等立体角/正交投影：均可用 OpenCV fisheye 模块标定，畸变系数会自动适配
- 已有其他格式标定结果：需转换为 OpenCV fisheye 格式，或自行编写 remap 查找表

### 8.4 计算性能

| 操作 | 分辨率 | 平台 | 典型耗时 | 说明 |
|------|--------|------|---------|------|
| `cv2.remap` (CV_16SC2) | 1920×1080 → 640×360 | x86 (i7) | ~0.5 ms | 定点查找表，cache 友好 |
| `cv2.remap` (CV_16SC2) | 1920×1080 → 640×360 | ARM (A76) | ~1.5 ms | NEON 自动向量化 |
| `cv2.remap` (CV_32FC1) | 1920×1080 → 640×360 | x86 (i7) | ~1.2 ms | 浮点表，精度更高但更慢 |

**建议**：
- 使用 `CV_16SC2` 定点格式（如 Step 3 所示），性能最优且精度足够
- 640×360 输出分辨率下 remap 耗时远小于模型推理（通常 > 10 ms），不是瓶颈
- 如需进一步优化，可考虑 OpenCV CUDA 版 `cv2.cuda.remap()` 或自定义 NEON kernel

### 8.5 在线标定与外参漂移

当前方案依赖离线标定 + 机械安装精度。生产级 ADAS 系统还需考虑：

**外参漂移来源**：
- 振动导致相机支架松动（长期）
- 碰撞或维修后未重新标定
- 挡风玻璃更换

**应对方案**（按复杂度递增）：

1. **定期手动校准**：使用 Section 5 的消失点十字标记，在保养时检查
2. **开机自检**：启动时自动检测消失点位置，偏差过大则告警
3. **在线外参估计**：运行时持续检测消失点并更新 R 矩阵中的 pitch/yaw 补偿。openpilot 的 `liveCalibration` 模块实现了类似功能，可参考其 EKF 方法

当前项目阶段建议先实施方案 1，在 Section 5 校准工具中增加偏差量化输出。

### 8.6 鱼眼镜头热漂移

车载环境温度范围大（-20°C ~ 80°C），鱼眼镜头（尤其塑料镜片）的畸变参数会随温度变化：

- **玻璃镜片**：热膨胀系数小，焦距漂移 < 0.1%，通常可忽略
- **塑料镜片**：焦距漂移可达 0.5%–1%，边缘畸变变化更大

**建议**：
- 在高温 (60°C) 和低温 (-10°C) 下各做一次标定，对比畸变参数差异
- 如差异导致 remap 后车道线位置偏差 > 5px (640×360)，考虑多温度标定或在线补偿
- 选型时优先选择玻璃镜片的鱼眼模组

### 8.7 适配不同相机规格

本方案假设鱼眼相机为 1920×1080、HFOV=120°。适配其他规格时：

| 参数变化 | 影响 | 处理方法 |
|---------|------|---------|
| 分辨率不同（如 1280×720） | f_fish 改变，需重新计算分辨率余量 | 重新标定，按 Section 2 方法验证分辨率充足 |
| HFOV > 120°（如 150°, 180°） | 70° 中心区域占比更小，源像素更充裕 | 正常适用，分辨率余量更大 |
| HFOV < 100° | 可能与 70° 裁剪区域接近，余量不足 | 需按 Section 2 重新分析，可能需缩小 CROP_HFOV |
| 非 16:9 宽高比 | cy 计算中主点位置不同 | 重新标定，K_model 的 cy 需从标定结果推算 |

**核心原则**：只要鱼眼相机标定内参 (K_fisheye, D) 准确，`cv2.fisheye.initUndistortRectifyMap` 会自动处理投影差异。K_model（目标虚拟针孔内参）始终保持不变——它由训练时的 crop 参数决定，与部署相机无关。
