# openpilot 在美国 Class 8 大卡车上的适用性分析

## 背景

openpilot 原生运行在 comma 3X 硬件上，设计目标是乘用车（轿车/SUV）。本文分析将 openpilot 感知管线（modeld + calibrationd）部署到美国 Class 8 半挂卡车（semi truck）时面临的技术差异与适配挑战。

## 安装高度差异

### 乘用车 vs 卡车

| 车型 | dashcam 离地高度 | 来源 |
|------|-----------------|------|
| 乘用车（轿车/SUV） | 1.1 ~ 1.3 m | comma 3X 典型安装 |
| Class 8 卡车（最低） | ~1.9 m | FHWA 驾驶员眼高 71.5 in |
| Class 8 卡车（平均） | ~2.4 m | FHWA 驾驶员眼高 93 in |
| Class 8 卡车（最高） | ~3.0 m | FHWA 驾驶员眼高 112.5 in |

> 数据来源：FHWA Recommended Guidelines（https://mutcd.fhwa.dot.gov/rpt/tcstoll/chapter514.htm）
> dashcam 安装在挡风玻璃上沿（后视镜附近），比驾驶员眼高略高约 0.1 ~ 0.15m。

卡车的 dashcam 安装高度约为乘用车的 **1.6x ~ 2.5x**，典型值约 2.0 倍。

### 高度差异的影响链

```
安装高度变化
  ├── 图像变换（warp matrix） → 不受影响（纯角度校正，无高度参数）
  ├── 在线标定（calibrationd）
  │     ├── HEIGHT_INIT = 1.22m → 初始偏差大，需要收敛时间
  │     └── 滑动窗口融合 → 最终会收敛到实际高度
  ├── UI 渲染（model_renderer）
  │     └── path_offset_z = height → 收敛前渲染位置偏移
  └── 模型推理（supercombo）
        └── 透视几何变化 → 核心风险，下文详述
```

## 高度对各模块的影响

### 1. 图像变换（warp matrix）—— 无影响

`get_warp_matrix()` 仅使用旋转角度（roll/pitch/yaw）和相机内参，不包含高度参数：

```python
# common/transformations/model.py:65-70
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
  calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
  device_from_calib = rot_from_euler(device_from_calib_euler)
  camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
  warp_matrix = camera_from_calib @ calib_from_model
  return warp_matrix
```

预计算的 `medmodel_frame_from_calib_frame` 和 `sbigmodel_frame_from_calib_frame` 均使用 `height=0`。结论：**无论相机安装多高，送入模型的图像裁剪/缩放逻辑完全一致。**

### 2. 在线标定（calibrationd）—— 可自适应，但有局限

calibrationd 从 modeld 的 `road_transform_trans[2]` 输出实时估计相机高度。

**能工作的部分**：
- 高度估计通过滑动窗口平滑（100 样本/block × 50 block），最终会收敛到实际安装高度
- 收敛后的高度值通过 `liveCalibration.height` 发布，UI 和规划模块可正确使用

**局限性**：
- **初始偏差大**：`HEIGHT_INIT = 1.22m`，卡车实际约 2.4m，偏差近 1 倍。冷启动期间渲染和规划使用错误高度
- **无突变检测**：rpy 有 spread 检测触发重标定（`MAX_ALLOWED_PITCH_SPREAD` / `MAX_ALLOWED_YAW_SPREAD`），但高度没有类似保护（代码中有 `TODO: add height spread check`）
- **无范围校验**：rpy 有 `PITCH_LIMITS` / `YAW_LIMITS` 做合法性检查，高度没有范围限制
- **收敛前提**：高度更新需要模型输出的 `road_transform_trans_std[2] < exp(-3.5) ≈ 0.03m`。在卡车视角下，模型对路面估计的置信度是否足够尚不确定

### 3. 模型推理（supercombo）—— 核心风险

这是最关键的适配挑战。supercombo 模型在大量乘用车视角（~1.2m 高度）的驾驶数据上训练，卡车视角（~2.4m）的透视几何有本质差异：

#### 3.1 地平线位置偏移

相机越高，地平线在图像中的位置越高（占比越大）。MEDMODEL 的 `cy = 47.6`（图像上方 19%），预期大部分画面是路面。在卡车高度下：

- 路面在图像中的占比增大
- 地平线上移，远处路面可见范围扩大
- 模型可能从未见过这种透视分布，预测质量不确定

#### 3.2 前车外观变化

从卡车高度俯视前车：
- 能看到前车车顶（乘用车视角看不到）
- 前车的尺度-距离关系不同（同距离下前车在画面中更小）
- lead detection 模型可能对这种视角的训练数据不足

#### 3.3 车道线透视比例

高度影响平行线的汇聚速度：
- 相机越高，车道线汇聚越慢（近处线间距更大）
- 模型学到的车道线宽度-距离映射可能不适用

#### 3.4 路面纹理/特征

卡车视角下路面纹理的透视变形模式不同，可能影响视觉里程计（`cameraOdometry`）的精度，进而影响标定收敛。

### 4. UI 渲染 —— 高度收敛前不准

`model_renderer.py` 使用 `liveCalibration.height[0]` 作为 Z 偏移量（`path_offset_z`）：

```python
# selfdrive/ui/onroad/model_renderer.py:100
self._path_offset_z = live_calib.height[0] if live_calib.height else HEIGHT_INIT[0]
```

在 `_map_line_to_polygon` 中，所有 3D 点的 Z 坐标都加上这个偏移。如果高度值不正确：
- 车道线和路径的渲染位置在屏幕上会偏移
- 前车标记的纵向位置不准确

calibrationd 收敛后此问题自动解决。

## 适配建议

### 短期（不改模型）

| 优先级 | 措施 | 说明 |
|--------|------|------|
| P0 | 修改 `HEIGHT_INIT` | 为卡车场景设置合理初始值（如 2.3m），减少冷启动偏差 |
| P0 | 添加高度范围校验 | 类似 `PITCH_LIMITS`，添加 `HEIGHT_LIMITS`（如 0.8 ~ 3.5m） |
| P1 | 添加高度 spread 检测 | 实现代码中的 TODO，检测高度突变并触发重标定 |
| P1 | 在 Carla 中验证 | 用 dashcam 工具设置卡车高度（`--camera-height 2.4`），评估模型输出质量 |
| P2 | 可配置 HEIGHT_INIT | 通过 Params 或命令行参数设置初始高度，支持不同车型 |

### 中期（模型层面）

| 措施 | 说明 |
|------|------|
| 收集卡车视角训练数据 | 在卡车上安装 comma 3X 或等效相机采集数据 |
| 数据增强 | 合成不同高度的视角变换，增加模型对高度变化的鲁棒性 |
| 高度感知训练 | 将相机高度作为模型输入条件，让模型学会适应不同安装高度 |

### 长期（架构层面）

| 措施 | 说明 |
|------|------|
| 高度纳入 warp 变换 | 当前 warp 只做角度校正。考虑将高度信息编码到图像变换中，补偿透视差异 |
| 多车型标定 profile | 为不同车辆类型（轿车/SUV/卡车）预设不同的标定参数初始值和范围 |

## 用 dashcam 工具验证

可以使用现有的 dashcam 工具在 Carla 中模拟卡车视角：

```bash
# 模拟卡车高度（2.4m），理想安装角度
python tools/dashcam/run.py --perfect-cam --camera-height 2.4 --high-quality

# 模拟卡车高度 + 安装偏差（5° 俯仰 + 3° 偏航）
python tools/dashcam/run.py --camera-height 2.4 --camera-pitch 5 --camera-yaw 3

# 在线标定模式，观察高度收敛过程
python tools/dashcam/run.py --online-calib --camera-height 2.4
```

观察要点：
- 车道线检测是否正常（绿色多边形是否贴合车道线）
- 前车检测是否准确（红色三角标记是否正确跟踪前车）
- calibrationd 的高度估计是否收敛到 2.4m 附近
- 模型输出的置信度（概率/标准差）是否合理

## 结论

openpilot 在卡车上的适配，**图像变换和标定系统在工程上是可行的**（warp 不含高度、calibrationd 可在线收敛），但 **模型层面存在根本性风险**（supercombo 未在卡车视角数据上训练）。建议先通过 Carla 仿真快速评估模型在高视点下的表现，再决定是否需要重新训练或微调模型。

## 参考

| 文件路径 | 内容 |
|---------|------|
| `common/transformations/model.py` | warp 矩阵计算，MEDMODEL/SBIGMODEL 参数 |
| `selfdrive/locationd/calibrationd.py` | 在线标定（rpy + height），HEIGHT_INIT 定义 |
| `selfdrive/modeld/modeld.py` | 模型推理主循环，warp 矩阵应用 |
| `selfdrive/ui/onroad/model_renderer.py` | UI 渲染，height 用作 path_offset_z |
| `docs/camera_selection_guide.md` | 物理相机选型要求 |
