# 基于 openpilot 真车数据的车道线标注方案

## 1. 问题定义

### 1.1 目标

从 openpilot 在真实车辆上采集的数据中，自动生成高质量的车道线标签，用于训练 supercombo 视觉模型。

### 1.2 约束条件

**可用传感器**（comma 3X 硬件）：

| 传感器 | 频率 | 精度 | 说明 |
|--------|------|------|------|
| 前向相机 (fcam) | 20Hz | 1928×1208, f=2648 | 窄角，模型主输入 |
| 广角相机 (ecam) | 20Hz | 1928×1208, f=567 | 鱼眼，模型辅助输入 |
| IMU 加速度计 | 104Hz | ±0.01 m/s² | 三轴 |
| IMU 陀螺仪 | 104Hz | ±0.01°/s | 三轴 |
| GPS | 1-10Hz | 水平 2-5m | u-blox GNSS |
| CAN 总线 | 100Hz | 车速 ±0.1 m/s | vEgo, 转向角等 |
| 雷达 | ~20Hz | 视车型而定 | 部分车型有 |

**不可用**：LiDAR、高精地图、人工标注

### 1.3 标签格式

与模型输出一致：4 条车道线，每条 33 个采样点（X_IDXS 距离处），每点 2 维 (y, z)，加 4 个存在概率。

---

## 2. 可用数据分析

### 2.1 模型自身输出

openpilot 在行驶中持续运行 supercombo 模型，其输出记录在日志中：

```
modelV2.laneLines[0:4]     — 4 条车道线的 (x, y, z) 轨迹，33 点
modelV2.laneLineProbs[0:4] — 存在概率
modelV2.laneLineStds[0:4]  — 不确定性
```

**关键观察**：从 Carla 评估结果看，模型在不同距离的精度差异巨大：

| 距离段 | MAE | 精度等级 |
|--------|-----|---------|
| 0-30m (near) | ~5cm | 高精度 |
| 30-60m (mid) | ~20cm | 中等精度 |
| 60-100m (far) | ~90cm | 低精度 |

这意味着**近距离预测是高质量的伪标签源**。

### 2.2 自车运动估计

来自 locationd 的 EKF 融合：

```
livePose.velocityDevice     — 设备坐标系速度 (m/s)，±0.05 m/s
livePose.angularVelocityDevice — 角速度 (rad/s)，±0.001 rad/s
livePose.orientationNED     — NED 姿态角
cameraOdometry.trans/rot    — 相机里程计（20Hz）
```

短时间（<5s）内的帧间变换精度可达 ±2cm 平移、±0.05° 旋转。

### 2.3 相机标定

```
liveCalibration.rpyCalib    — [roll, pitch, yaw] 相机外参
liveCalibration.height      — 相机离地高度
liveCalibration.calStatus   — 标定状态
```

标定收敛后（validBlocks ≥ 5），pitch/yaw 精度约 ±0.05°。

---

## 3. 核心方案：时间累积 + 近距离回溯

### 3.1 核心思想

车辆以 v m/s 行驶时，每秒前进 v 米。当前帧 100m 处的车道线点，在 100/v 秒后变成 0m 处的点。**远处的不确定性在车辆驶近后变为近处的高精度观测**。

```
时间线：t=0                t=3s               t=6s
         ├─ 观测 100m 处 ──┤─ 观测 10m 处 ───┤─ 已驶过
         │  MAE ~90cm       │  MAE ~5cm        │
         │  (低精度)        │  (高精度)        │
```

因此，**同一个世界点**可以在不同时刻被多次观测，其中最精确的是距离 <30m 时的观测。

### 3.2 算法流程

```
输入：连续 N 秒（如 10s）的日志数据
输出：每帧的高精度车道线标签

Step 1: 提取原始数据
  对每帧 t，提取：
  - model_lines[t]: 模型输出的 4 条车道线 (4×33×3)
  - model_probs[t]: 存在概率 (4,)
  - model_stds[t]: 不确定性 (4,)
  - pose[t]: livePose 位姿
  - calib[t]: liveCalibration 外参

Step 2: 构建世界坐标系
  选择参考帧 t_ref（通常取窗口中点）
  对每帧 t，计算 t_ref → t 的刚体变换 T(t_ref, t)
  使用 livePose 的速度和角速度积分：
    ΔR = ∫ ω dt    (旋转)
    Δp = ∫ v dt    (平移)

Step 3: 逆投影到世界坐标
  对每帧 t 的每个车道线点 p_calib = (x, y, z)：
    p_world = T(t_ref, t) × p_calib

Step 4: 距离加权累积
  对世界坐标中的每个区域，收集所有帧的观测
  根据观测时的距离分配权重：
    w(x) = 1/σ²(x)   其中 σ(x) 随距离增大
  加权平均得到高精度的世界坐标车道线

Step 5: 反投影到每帧
  将累积后的世界坐标车道线反变换到每帧的标定坐标系
  在 X_IDXS 处插值，得到标签
```

### 3.3 距离-权重函数

基于 Carla 评估得到的经验精度模型：

```python
def observation_weight(x_distance):
    """根据观测距离计算权重，距离越近权重越高"""
    if x_distance <= 30:
        sigma = 0.05  # 5cm
    elif x_distance <= 60:
        sigma = 0.05 + (x_distance - 30) / 30 * 0.15  # 5cm → 20cm
    else:
        sigma = 0.20 + (x_distance - 60) / 40 * 0.70  # 20cm → 90cm
    return 1.0 / (sigma ** 2)
```

### 3.4 帧间变换的精确计算

```python
def compute_frame_transform(pose_t, pose_ref, dt):
    """计算从参考帧到目标帧的刚体变换"""
    # 使用 livePose 的速度和角速度积分
    v = pose_t.velocityDevice  # [vx, vy, vz] m/s
    omega = pose_t.angularVelocityDevice  # [wx, wy, wz] rad/s

    # 旋转增量（小角度近似或 Rodrigues）
    dR = rot_from_euler(omega * dt)

    # 平移增量（设备坐标系）
    dp = v * dt

    return dR, dp
```

对于更高精度，使用 `cameraOdometry` 的逐帧累积（20Hz），而非积分。

---

## 4. 增强策略

### 4.1 策略 A：近距离回溯标注

**原理**：利用未来帧的近距离高精度观测作为当前帧远距离位置的标签。

```
帧 t=0: 模型预测 x=90m 处车道线 y=-1.8m (MAE ~90cm)
帧 t=3s: 车辆前进了 ~30m，同一点现在在 x=60m 处
帧 t=6s: 车辆又前进了 ~30m，同一点现在在 x=30m 处
          模型预测该点 y=-1.82m (MAE ~5cm) ← 高精度标签

将 t=6s 的高精度观测通过帧间变换回溯到 t=0 的坐标系，
作为 t=0 帧 x=90m 处的标签。
```

**实现**：

```python
def retrospective_labeling(frames, window_sec=10):
    """近距离回溯标注"""
    labels = {}
    for t_target in frames:
        accumulated_points = []  # (world_x, world_y, world_z, weight)

        for t_obs in frames:
            if abs(t_obs - t_target) > window_sec:
                continue

            T = get_transform(t_target, t_obs)  # t_obs → t_target 坐标变换
            lines = frames[t_obs].model_lines
            probs = frames[t_obs].model_probs

            for line_idx in range(4):
                if probs[line_idx] < 0.5:
                    continue
                for pt_idx in range(33):
                    x_obs = X_IDXS[pt_idx]  # 观测时的距离
                    if np.isnan(lines[line_idx][pt_idx, 1]):
                        continue

                    # 将观测点变换到目标帧坐标系
                    pt_calib = lines[line_idx][pt_idx]  # (x, y, z)
                    pt_target = T @ pt_calib

                    w = observation_weight(x_obs)  # 近距离权重高
                    accumulated_points.append((line_idx, pt_target, w))

        # 对每条线，在 X_IDXS 处加权插值
        labels[t_target] = weighted_interpolate(accumulated_points)

    return labels
```

### 4.2 策略 B：车辆轨迹作为车道中心先验

**原理**：在正常行驶中，车辆大致沿车道中心线行驶。将 GPS+IMU 融合的平滑轨迹作为车道中心的强先验。

```python
def trajectory_lane_prior(poses, lane_width=3.7):
    """从车辆轨迹推导车道中心线和边界"""
    # 1. 平滑车辆轨迹（Savitzky-Golay 或样条滤波）
    trajectory = smooth_trajectory(poses)

    # 2. 计算轨迹的法向量（指向右侧）
    normals = compute_normals(trajectory)

    # 3. 车道边界 = 轨迹 ± 半车道宽度 × 法向量
    left_boundary = trajectory - (lane_width / 2) * normals
    right_boundary = trajectory + (lane_width / 2) * normals

    return left_boundary, right_boundary
```

**限制**：
- 假设车辆在车道中心行驶（变道、靠边停车时不成立）
- 车道宽度需要估计（可用模型输出的 lane_width 作为初始值）
- 精度受 GPS 限制（2-5m），仅作为粗略先验

**改进**：结合模型输出的近距离车道线（精度 ~5cm），校正轨迹到实际车道中心的偏移量：

```python
# 模型近距离输出：左线 y_left ≈ -1.85m，右线 y_right ≈ 1.85m
# 车道中心偏移 = (y_left + y_right) / 2
# 如果 offset ≈ 0.1m，说明车辆偏右 0.1m
center_offset = (model_left_y_near + model_right_y_near) / 2
# 修正轨迹
corrected_trajectory = trajectory + center_offset * normals
```

### 4.3 策略 C：图像空间车道标线检测

**原理**：车道线在图像中通常是高对比度的白色/黄色标线。使用经典 CV 方法在图像空间检测标线，然后提升到 3D。

```python
def image_lane_detection(image, intrinsics, rpyCalib, height):
    """图像空间车道标线检测 + 3D 提升"""
    # 1. 颜色空间转换和阈值
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white_mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] > 200)
    yellow_mask = (hsv[:,:,0] > 15) & (hsv[:,:,0] < 35) & (hsv[:,:,1] > 100)
    lane_mask = white_mask | yellow_mask

    # 2. Canny 边缘检测 + Hough 直线
    edges = cv2.Canny(lane_mask.astype(np.uint8) * 255, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=30)

    # 3. 像素坐标 → 标定坐标系 3D 点
    # 假设路面为平面 z = height（相机高度）
    # 反投影：对每个像素 (u, v)
    #   视图坐标 = K^-1 @ [u, v, 1]
    #   标定坐标 = device_from_view @ view_pt
    #   在路面约束 z = height 下解出 (x, y)
    lane_3d = unproject_to_road_plane(lane_pixels, intrinsics, rpyCalib, height)

    return lane_3d
```

**优势**：直接从图像特征提取，不依赖模型预测
**限制**：
- 磨损、被遮挡、无标线的路段无法检测
- 需要精确的路面平面假设
- 计算量大（需处理高分辨率图像）

**适用场景**：作为辅助验证手段，与模型预测交叉校验。

### 4.4 策略 D：多车次地图累积

**原理**：同一路段被多辆车多次行驶，每次的车道线观测可以累积到一个共享地图中。

```python
def multi_trip_accumulation(trips_on_same_road):
    """多车次同一路段的车道线累积"""
    # 1. 使用 GPS 将各车次对齐到同一全局坐标系
    # 2. 每车次贡献近距离（<30m）的高精度车道线观测
    # 3. 加权平均（权重 = 1/σ² × 标定质量 × 速度稳定性）
    # 4. 生成该路段的"地图级"车道线标签
    # 5. 后续车次经过该路段时，地图标签可直接使用
```

**优势**：多次观测显著降低噪声，接近"HD 地图"精度
**限制**：需要 GPS 对齐（精度 2-5m），需要路段匹配算法

---

## 5. 推荐实现方案

### 5.1 总体架构

```
┌────────────────────────────────────────────────────┐
│ 阶段 1: 数据准备                                    │
│                                                    │
│ rlog.zst → 提取每帧:                                │
│   • modelV2.laneLines + probs + stds               │
│   • livePose (velocity, angularVelocity)           │
│   • liveCalibration (rpyCalib, height, calStatus)  │
│   • carState.vEgo                                  │
│   • cameraOdometry (trans, rot)                    │
└────────────────────┬───────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────┐
│ 阶段 2: 质量筛选                                    │
│                                                    │
│ 跳过以下帧:                                         │
│   • calStatus ≠ calibrated                         │
│   • vEgo < 5 m/s (低速)                            │
│   • |yaw_rate| > 0.5 rad/s (急转弯)                │
│   • laneLineProbs < 0.3 (低检测置信度)              │
│   • posenetOK = false                              │
│   • 模型输出存在异常大的 std                         │
└────────────────────┬───────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────┐
│ 阶段 3: 帧间变换计算                                 │
│                                                    │
│ 使用 cameraOdometry 逐帧累积:                        │
│   T(t_ref, t) = ∏ ΔT(t_i, t_{i+1})               │
│ 每 50ms 一次（20Hz），5s 窗口内精度 ±5cm             │
└────────────────────┬───────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────┐
│ 阶段 4: 近距离回溯累积                               │
│                                                    │
│ 对每帧 t_target:                                    │
│   • 收集 [t-5s, t+5s] 窗口内所有帧的车道线观测      │
│   • 通过 T(t_target, t_obs) 变换到目标帧坐标系       │
│   • 按观测距离加权 w = 1/σ²(x_obs)                  │
│   • 在 X_IDXS 处加权插值                            │
└────────────────────┬───────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────┐
│ 阶段 5: 时间一致性滤波                               │
│                                                    │
│ 对累积后的标签做时间平滑:                             │
│   • 相邻帧的车道线标签应平滑变化                      │
│   • 检测跳变并标记为低质量                            │
│   • 可选: 对标签做 Savitzky-Golay 时间滤波           │
└────────────────────┬───────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────┐
│ 阶段 6: 输出标签                                     │
│                                                    │
│ 每帧输出:                                           │
│   label_lines: (4, 33, 3)  — X_IDXS 处 (x, y, z)  │
│   label_probs: (4,)        — 存在概率                │
│   label_quality: float     — 标签质量分              │
└────────────────────────────────────────────────────┘
```

### 5.2 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 累积窗口 | ±5 秒 | 30 m/s 速度下覆盖 ~300m |
| 最低车速 | 5 m/s | 低速里程计精度下降 |
| 最大偏航角速度 | 0.5 rad/s | 弯道里累积精度受限 |
| 最低标定状态 | calibrated | 未标定帧不使用 |
| 近距离信赖范围 | 0-30m | MAE < 5cm |
| 中距离参考范围 | 30-60m | MAE < 20cm |
| 远距离限制 | >60m | 主要依赖累积 |

### 5.3 标签质量评分

```python
def compute_label_quality(frame_data, accumulated_label):
    """计算单帧标签的质量分数 [0, 1]"""
    score = 1.0

    # 标定质量
    if frame_data.cal_status != 'calibrated':
        score *= 0.0
    elif frame_data.cal_spread > 0.5:
        score *= 0.5

    # 速度稳定性
    if frame_data.v_ego < 5:
        score *= 0.3
    elif frame_data.v_ego < 15:
        score *= 0.7

    # 累积观测数量
    n_obs = accumulated_label.num_observations
    score *= min(n_obs / 100, 1.0)  # 100 次观测满分

    # 车道线检测置信度
    avg_prob = np.mean([p for p in frame_data.lane_probs if p > 0.3])
    score *= avg_prob

    # 时间一致性（与相邻帧标签的差异）
    temporal_jitter = accumulated_label.temporal_consistency
    score *= max(0, 1.0 - temporal_jitter / 0.5)

    return score
```

---

## 6. 精度分析

### 6.1 各距离段的预期标注精度

**单帧模型输出** vs **累积标签**（10s 窗口，30 m/s 车速）：

| 距离段 | 单帧 MAE | 累积观测次数 | 累积标签预期 MAE |
|--------|---------|------------|-----------------|
| 0-30m | 5cm | 1 帧（已经足够精确）| ~5cm |
| 30-60m | 20cm | ~6 帧（来自不同距离的观测）| ~8cm |
| 60-100m | 90cm | ~10+ 帧（包含多次近距离回溯）| ~15cm |
| 100-150m | >1m | ~15+ 帧 | ~25cm |

**精度提升来源**：
- **近距离回溯**：100m 处的点在 ~3s 后变为近距离观测（5cm 精度）
- **多帧平均**：N 次独立观测，噪声降低 √N 倍
- **距离加权**：近距离观测权重远高于远距离

### 6.2 误差来源与控制

| 误差来源 | 量级 | 控制方法 |
|----------|------|---------|
| 帧间变换累积漂移 | ~2cm/s | 限制窗口 ≤10s，用 GPS 定期校正 |
| 相机标定误差 | pitch ±0.05° | 仅使用标定收敛后的帧 |
| 模型预测噪声 | 距离相关 | 距离加权累积 |
| 路面非平面 | 坡道/起伏 | 使用 z 坐标，不假设平面 |
| 车道变更 | 标签跳变 | desire 信号检测 + 跳变过滤 |
| GPS 多径/遮挡 | 2-20m | 仅用于多车次对齐，不用于帧间 |

### 6.3 与其他标注方法的精度对比

| 方法 | 近距离(0-30m) | 远距离(60-100m) | 成本 |
|------|-------------|----------------|------|
| 人工标注（2D） | ~10px (~15cm) | ~5px (~50cm) | 高 |
| LiDAR 点云标注 | ~2cm | ~5cm | 极高 |
| HD Map | ~10cm | ~10cm | 极高 |
| **时间累积法（本方案）** | **~5cm** | **~15cm** | **零** |
| 单帧模型输出 | ~5cm | ~90cm | 零 |

---

## 7. 实现注意事项

### 7.1 帧间变换的选择

推荐使用 `cameraOdometry`（20Hz 视觉里程计）而非 `livePose`（EKF 融合）：

- **cameraOdometry** 直接来自模型输出的帧间运动，与模型输入的图像严格对齐
- **livePose** 经过 EKF 平滑，时间上存在滤波延迟
- 短时间窗口内（<10s），视觉里程计的漂移可接受

```python
# 使用 cameraOdometry 逐帧累积变换
def accumulate_transforms(cam_odom_msgs):
    T_cumulative = np.eye(4)
    transforms = {t0: np.eye(4)}

    for msg in cam_odom_msgs:
        dt = 1.0 / 20.0  # 50ms
        trans = np.array(msg.trans) * dt  # 帧间平移
        rot = np.array(msg.rot) * dt      # 帧间旋转
        dT = rigid_transform(rot, trans)
        T_cumulative = T_cumulative @ dT
        transforms[msg.t] = T_cumulative.copy()

    return transforms
```

### 7.2 弯道处理

弯道中帧间变换的旋转分量较大，累积误差增长更快：

```python
def curve_quality_factor(yaw_rate, v_ego):
    """弯道的标注质量衰减因子"""
    radius = v_ego / max(abs(yaw_rate), 1e-6)
    if radius > 500:      # 直道
        return 1.0
    elif radius > 100:    # 缓弯
        return 0.8
    elif radius > 50:     # 中等弯道
        return 0.5
    else:                 # 急弯
        return 0.2        # 显著降低权重
```

### 7.3 变道检测与过滤

变道过程中车道线标签会跳变，需要检测并特殊处理：

```python
def detect_lane_change(model_outputs, desire_state):
    """检测变道，标记变道帧为低质量"""
    is_lane_change = desire_state in [
        Desire.laneChangeLeft, Desire.laneChangeRight
    ]
    # 也可以通过车道线 y 值的突变检测
    left_y_jump = abs(diff(model_left_y_near))
    is_jump = left_y_jump > 0.5  # 0.5m 跳变
    return is_lane_change or is_jump
```

### 7.4 隧道和路口处理

- **隧道/遮挡**：GPS 不可用，但短时间视觉里程计仍准确
- **路口**：车道线消失，应跳过。检测方法：laneLineProbs 同时降至 <0.3
- **匝道**：车道合并/分离，标签不稳定，应降低质量分

### 7.5 标签存储格式

```python
@dataclass
class LaneLineLabel:
    frame_id: int
    timestamp: float

    # 4 条车道线标签 (标定坐标系)
    lines: np.ndarray       # shape (4, 33, 3)  — (x, y, z) at X_IDXS
    probs: np.ndarray       # shape (4,)        — 存在概率
    quality: float          # 标签质量分 [0, 1]

    # 元数据
    v_ego: float            # 车速
    cal_status: str         # 标定状态
    num_observations: int   # 累积观测次数
    window_sec: float       # 使用的窗口大小
```

---

## 8. 与训练流程的集成

### 8.1 离线标签生成管线

```bash
# 1. 下载 route 日志
python tools/lib/logreader.py "route_id"

# 2. 运行标签生成
python tools/labeling/lane_label_generator.py \
    --route "dongle|timestamp" \
    --window 10.0 \
    --min-speed 5.0 \
    --output labels/

# 3. 质量过滤
python tools/labeling/filter_labels.py \
    --input labels/ \
    --min-quality 0.6 \
    --output filtered_labels/
```

### 8.2 训练时标签使用

```python
def load_training_sample(segment_path, frame_idx):
    """加载训练样本：图像 + 标签"""
    # 图像
    fr = FrameReader(segment_path + "/fcamera.hevc")
    image = fr.get(frame_idx)  # 1928×1208 RGB

    # 标签
    label = load_label(segment_path + "/lane_labels.npz", frame_idx)

    # 只使用高质量标签
    if label.quality < 0.6:
        return None

    # 转换为训练格式: (y, z) at X_IDXS
    target_lines = label.lines[:, :, 1:3]  # (4, 33, 2)
    target_probs = label.probs              # (4,)

    return image, target_lines, target_probs
```

---

## 9. 总结

### 9.1 方案优势

1. **零标注成本**：完全自动化，利用模型自身预测 + 车辆运动
2. **高精度**：近距离回溯 + 多帧累积，远距离精度从 ~90cm 提升到 ~15cm
3. **大规模**：每辆 openpilot 车辆的每次行驶都自动产生标签
4. **自迭代**：更好的模型 → 更好的标签 → 更好的下一代模型

### 9.2 局限性

1. **首代模型问题**：需要一个初始模型来提供基础预测（冷启动问题）
2. **帧间变换漂移**：>10s 窗口时累积误差显著
3. **弯道精度下降**：旋转累积误差更大
4. **无标线路段**：无法标注没有物理标线的道路（如土路）

### 9.3 推荐的实施优先级

| 优先级 | 策略 | 预期精度提升 | 实现复杂度 |
|--------|------|------------|-----------|
| **P0** | 近距离回溯累积 | 远距离 90cm → 15cm | 中 |
| **P1** | 时间一致性滤波 | 减少跳变噪声 | 低 |
| **P2** | 车辆轨迹先验 | 提供车道中心约束 | 低 |
| **P3** | 图像空间检测辅助 | 交叉验证 | 高 |
| **P4** | 多车次地图累积 | 地图级精度 (~5cm) | 高 |
