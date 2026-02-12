# calibrationd 工作原理与输出数据用途

## 1. 概述

`calibrationd`（`selfdrive/locationd/calibrationd.py`）是 openpilot 的**在线相机外参标定进程**。它持续估计摄像头相对于车辆的安装姿态（俯仰角、偏航角）和离地高度，为视觉模型和控制系统提供坐标变换基础。

### 为什么需要标定？

comma 3X 设备通过挡风玻璃支架安装在车内，每次安装的角度都会有细微差异。视觉神经网络（supercombo model）在一个**标准标定坐标系**下训练，如果不补偿实际安装角度偏差，模型的车道线、路径预测、前车检测等所有输出都会产生系统性偏移。

calibrationd 的作用就是在线估计这个偏差角度，让系统自动适应不同的安装姿态。

### 核心输入输出

```
输入：
  cameraOdometry  ← modeld（视觉模型输出的自运动估计）
  carState         ← 车辆状态（车速）

输出：
  liveCalibration  → 几乎所有子系统
```

## 2. 坐标系定义

理解 calibrationd 需要先理解 openpilot 中的坐标系体系（`common/transformations/camera.py`）：

### 2.1 设备坐标系（Device Frame）

以设备（comma 3X）为中心：
- **x** → 向前（车头方向）
- **y** → 向右
- **z** → 向下

### 2.2 标定坐标系（Calibration Frame）

与设备坐标系形状相同，但代表**理想安装姿态**下的设备坐标系。当 `rpyCalib = [0, 0, 0]` 时，标定坐标系与设备坐标系完全重合，表示设备完美水平安装。

实际中 `rpyCalib` 不为零，表示设备相对于理想位置有旋转偏移：
- `rpyCalib[0]`（roll）：始终近似为 0（不估计横滚）
- `rpyCalib[1]`（pitch）：设备俯仰角，正值表示向下倾斜
- `rpyCalib[2]`（yaw）：设备偏航角，正值表示向左偏转

### 2.3 视图坐标系（View Frame）

相机/渲染使用的坐标系：
- **x** → 向右
- **y** → 向下
- **z** → 向前

设备坐标系到视图坐标系的变换是一个固定的坐标轴重排（`camera.py:75-80`）：

```python
device_frame_from_view_frame = [[0, 0, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]]
view_frame_from_device_frame = device_frame_from_view_frame.T
```

### 2.4 模型坐标系（Model Frame）

神经网络输入图像的坐标系，由模型的虚拟内参定义。模型在训练时使用标定坐标系，因此**模型坐标系通过固定的虚拟内参与标定坐标系关联**。

### 2.5 坐标变换链

从模型像素到实际相机像素的完整变换链：

```
模型像素 → (calib_from_model) → 标定坐标系 → (device_from_calib) → 设备坐标系
         → (view_from_device) → 视图坐标系 → (K, 相机内参) → 相机像素
```

`get_warp_matrix()` 将上述链条组合为一个 3x3 单应性矩阵（`common/transformations/model.py:65-70`）：

```python
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix
```

这个 `warp_matrix` 将原始相机图像透视变换为模型期望的标准视角图像。

## 3. 标定算法

### 3.1 核心原理

calibrationd 利用一个简单但有效的几何关系：**在直行且匀速时，车辆的速度方向应该指向正前方**。如果摄像头有安装偏移，视觉模型估计的速度方向就会偏离正前方，这个偏移量就反映了摄像头的安装角度。

具体地，从摄像头里程计的平移向量 `trans = [tx, ty, tz]` 反推俯仰和偏航：

```python
observed_rpy = [0,
                -arctan2(tz, tx),    # pitch：前进方向的俯仰角
                 arctan2(ty, tx)]    # yaw：前进方向的偏航角
```

### 3.2 条件门控

并非所有时刻都适合更新标定。只有在满足以下**全部条件**时才会进行一次标定更新（`calibrationd.py:229-244`）：

| 条件 | 阈值 | 原因 |
|------|------|------|
| 车速足够高 | `v_ego > 15 MPH`（约 6.7 m/s） | 低速时视觉里程计噪声大 |
| 视觉速度足够大 | `trans[0] > MIN_SPEED_FILTER` | 确认视觉模型也检测到运动 |
| 偏航角速率小 | `abs(rot[2]) < 2°/s` | 只在直行时更新，避免弯道噪声 |
| 方向角不确定度低 | `arctan2(trans_std[1], trans[0]) < 0.25°` | 模型对方向角有足够置信度 |
| 高度不确定度低 | `road_transform_trans_std[2] < exp(-3.5)` | 高度估计足够稳定 |

在标定初期（`valid_blocks < INPUTS_NEEDED`），方向角和高度的不确定度条件会被放宽，以加速冷启动收敛。

### 3.3 Block 滑动窗口机制

calibrationd 不直接对单次观测做平均，而是使用一个**分块滑动窗口**的两级平滑结构：

**第一级：Block 内线性加权平均**

每个 block 包含 `BLOCK_SIZE = 100` 个有效样本。在一个 block 内，新旧样本的融合使用线性衰减权重：

```python
def moving_avg_with_linear_decay(prev_mean, new_val, idx, block_size):
    return (idx * prev_mean + (block_size - idx) * new_val) / block_size
```

越新的样本权重越大，使 block 内的估计更偏向最近的观测。

**第二级：跨 Block 均值**

维护一个大小为 `INPUTS_WANTED = 50` 的循环缓冲区存储历史 block 结果。最终的 rpy 取所有有效 block 的均值：

```python
self.rpy = np.mean(rpys[valid_idxs], axis=0)
```

### 3.4 标定状态机

```
                     blocks < 5
              ┌──────────────────────┐
              │                      │
              ▼                      │
         uncalibrated ───────────────┘
              │
              │ blocks >= 5 且 rpy 在有效范围内
              ▼
         calibrated ──────────────────┐
              │                       │
              │ pitch 散布 > 4° 或     │
              │ yaw 散布 > 2°         │
              ▼                       │
         recalibrating ───────────────┘
              │                  blocks >= 5 且散布恢复
              │
              │ blocks >= 5 但 rpy 超出范围
              ▼
           invalid
```

**状态定义**（`cereal/log.capnp:795-800`）：

| 状态 | 含义 | 触发条件 |
|------|------|----------|
| `uncalibrated` | 未校准 | `valid_blocks < 5` |
| `calibrated` | 已校准 | `valid_blocks >= 5` 且 rpy 在允许范围内 |
| `invalid` | 校准无效 | `valid_blocks >= 5` 但 rpy 超出允许范围 |
| `recalibrating` | 重新校准 | 检测到散布过大（可能安装位置变化） |

**有效 rpy 范围**（`calibrationd.py:56-59`）：

| 参数 | 最小值 | 最大值 | 说明 |
|------|--------|--------|------|
| Pitch | -5.2° | 9.7° | 适应不同的挡风玻璃倾斜角度 |
| Yaw | -3.96° | 3.96° | 适应轻微的左右偏转 |

### 3.5 平滑过渡

当检测到散布过大需要重新校准时，calibrationd 不会立刻跳变到新值，而是通过 `old_rpy_weight` 做平滑过渡（`SMOOTH_CYCLES = 10` 帧）：

```python
def get_smooth_rpy(self):
    if self.old_rpy_weight > 0:
        return self.old_rpy_weight * self.old_rpy + (1.0 - self.old_rpy_weight) * self.rpy
    else:
        return self.rpy
```

这避免了标定值的突变导致视觉模型输入图像的突变，保护控制回路的稳定性。

### 3.6 持久化

标定结果定期写入 `Params("CalibrationParams")`，设备重启后能从上次的标定状态恢复，无需从头开始。写入频率约为每 `INPUTS_WANTED / 5 = 10` 个 block 写一次。

## 4. 输出消息：liveCalibration

### 4.1 消息结构

```capnp
struct LiveCalibrationData {
  calStatus @11 :Status;              # 标定状态枚举
  calPerc @3 :Int8;                   # 标定进度百分比 (0-100%)
  validBlocks @9 :Int32;              # 有效 block 计数

  rpyCalib @7 :List(Float32);         # [roll, pitch, yaw] 设备安装角（弧度）
  rpyCalibSpread @8 :List(Float32);   # [roll, pitch, yaw] 散布（弧度）
  wideFromDeviceEuler @10 :List(Float32);  # 广角相机相对设备的欧拉角
  height @12 :List(Float32);          # 相机离路面高度（米）

  enum Status {
    uncalibrated @0;
    calibrated @1;
    invalid @2;
    recalibrating @3;
  }
}
```

### 4.2 发布频率

`liveCalibration` 以约 **4 Hz** 发布（`cameraOdometry` 频率为 20 Hz，每 5 帧发一次）。

### 4.3 典型值

| 字段 | 典型值 | 说明 |
|------|--------|------|
| `rpyCalib` | `[0, 0.06, -0.01]` | pitch ≈ 3.4°, yaw ≈ -0.6° |
| `height` | `[1.22]` | 相机离地 1.22m |
| `wideFromDeviceEuler` | `[0, 0, 0]` | 广角与窄角近乎对齐 |
| `calPerc` | `100` | 标定完成 |
| `validBlocks` | `20` | 已积累 20 个有效 block |

## 5. 输出数据的用途

### 5.1 modeld — 图像透视变换（最核心用途）

**文件**：`selfdrive/modeld/modeld.py:423-431`

这是 `rpyCalib` 最关键的用途。modeld 在每一帧图像送入神经网络之前，根据当前标定参数计算透视变换矩阵，将原始相机图像变换到模型训练时使用的标准视角：

```python
device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib)
model_transform_main = get_warp_matrix(device_from_calib_euler, intrinsics, False)
```

变换通过 OpenCL 的 `warpPerspective` 内核在 GPU 上执行双线性插值重采样。

**如果没有这个变换**：设备向下倾斜 5° 安装时，模型"看到"的路面会偏高，导致车道线检测向上偏移、路径规划错误、前车距离估计不准。

### 5.2 locationd — 摄像头里程计坐标变换

**文件**：`selfdrive/locationd/locationd.py:148-184`

locationd 的卡尔曼滤波器在**设备坐标系**下工作，而摄像头里程计 `cameraOdometry` 的 `trans` 和 `rot` 来自视觉模型，是在**标定坐标系**下的。locationd 需要用 `rpyCalib` 将其变换到设备坐标系：

```python
device_from_calib = rot_from_euler(rpyCalib)
rot_device = device_from_calib @ rot_calib
trans_device = device_from_calib @ trans_calib
```

### 5.3 controlsd — 姿态与角速率校准

**文件**：`selfdrive/controls/controlsd.py:63-64`

controlsd 通过 `PoseCalibrator`（`selfdrive/locationd/helpers.py:155-183`）将 EKF 输出的设备坐标系姿态变换到标定坐标系，然后下发给车辆控制器：

```python
calibrated_pose = pose_calibrator.build_calibrated_pose(device_pose)
CC.orientationNED = calibrated_pose.orientation.xyz.tolist()
CC.angularVelocity = calibrated_pose.angular_velocity.xyz.tolist()
```

`PoseCalibrator` 的核心变换：

```python
def feed_live_calib(self, live_calib):
    device_from_calib = rot_from_euler(live_calib.rpyCalib)
    self.calib_from_device = device_from_calib.T   # 逆变换
```

### 5.4 torqued — 横向扭矩标定

**文件**：`selfdrive/locationd/torqued.py:177-197`

torqued 使用校准后的角速率和 roll 角计算侧向加速度，用于在线拟合扭矩模型：

```python
lateral_acc = (v_ego * yaw_rate) - (sin(roll) * 9.81)
```

如果标定不准，`yaw_rate` 会有系统性偏差，导致扭矩模型拟合错误，影响横向控制精度。

### 5.5 dmonitoringmodeld — 驾驶员监控模型

**文件**：`selfdrive/modeld/dmonitoringmodeld.py:137-141`

驾驶员监控模型接收 `rpyCalib` 作为额外输入，用于补偿设备安装角度对人脸检测和视线方向估计的影响：

```python
calib[:] = np.array(sm["liveCalibration"].rpyCalib)
model_output = model.run(buf, calib, model_transform)
```

### 5.6 dmonitoringd — 人脸朝向校正

**文件**：`selfdrive/monitoring/helpers.py:129-145`

驾驶员监控状态机从人脸朝向中减去标定角度，得到人脸相对于车辆（而非设备）的真实朝向：

```python
pitch -= rpy_calib[1]
yaw -= rpy_calib[2]
```

### 5.7 lagd / paramsd — 参数学习

**文件**：`selfdrive/locationd/lagd.py:246-247`、`selfdrive/locationd/paramsd.py:114-115`

横向延迟估计器（lagd）和参数学习器（paramsd，估计转向比和轮胎刚度）都通过 `PoseCalibrator` 使用标定数据，确保姿态反馈在正确的坐标系下。

### 5.8 selfdrived — 告警生成

**文件**：`selfdrive/selfdrived/events.py:257-311`

selfdrived 根据标定状态生成用户告警：

| 标定状态 | 告警 | 表现 |
|----------|------|------|
| `uncalibrated` | "Calibrating: XX%" | 提示用户以 >15 MPH 直行 |
| `recalibrating` | "Recalibrating: XX%" | 检测到安装变化，重新校准 |
| `invalid` | "Calibration Invalid" | 显示当前 pitch/yaw 角度，提示重新安装设备 |

### 5.9 UI — 增强现实渲染

**文件**：`selfdrive/ui/onroad/augmented_road_view.py:138-159`

UI 层使用标定参数将 3D 世界坐标（车道线、路径、前车位置）投影到 2D 屏幕上。没有正确的标定，增强现实叠加层会与实际道路图像错位。

```python
device_from_calib = rot_from_euler(calib.rpyCalib)
view_from_calib = view_frame_from_device_frame @ device_from_calib
```

**`height` 字段**用于 `model_renderer.py` 中的路径渲染，确定 3D 点到 2D 屏幕的投影高度偏移。

### 5.10 UI 设置页面 — 标定角度显示

**文件**：`selfdrive/ui/layouts/settings/device.py:128-133`

设置页面展示当前标定角度，帮助用户判断设备安装是否合理：

```
"Your device is pointed 3.4° down and 0.6° left."
```

## 6. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│ modeld (视觉神经网络, 20 Hz)                                         │
│                                                                     │
│  原始图像 → warp(rpyCalib) → 标准视角图像 → supercombo model          │
│                                │                                    │
│                                ├─→ cameraOdometry (trans, rot, std) │
│                                ├─→ modelV2 (车道线, 路径, 前车)       │
│                                └─→ road_transform (路面高度)          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    cameraOdometry + carState(v_ego)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ calibrationd (标定进程, 4 Hz 输出)                                    │
│                                                                     │
│  条件门控：v_ego > 15 MPH, yaw_rate < 2°/s, std 足够小               │
│       ↓                                                             │
│  observed_rpy = [0, -atan2(tz/tx), atan2(ty/tx)]                    │
│       ↓                                                             │
│  Block 滑动窗口（100 样本/block, 50 block 缓冲区）                    │
│       ↓                                                             │
│  状态机：uncalibrated → calibrated → recalibrating / invalid         │
│       ↓                                                             │
│  → 发布 liveCalibration                                              │
│  → 持久化到 Params("CalibrationParams")                              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                   liveCalibration
                            │
    ┌───────────────────────┼──────────────────────────────┐
    │                       │                              │
    ▼                       ▼                              ▼
┌──────────┐       ┌───────────────┐              ┌───────────────┐
│ modeld   │       │ locationd     │              │ controlsd     │
│ 图像变换  │       │ 里程计坐标变换 │              │ 姿态/角速率校准 │
└──────────┘       └───────────────┘              └───────────────┘
    │                       │                              │
    ▼                       ▼                              ▼
┌──────────┐       ┌───────────────┐              ┌───────────────┐
│ dmonitor │       │ torqued       │              │ lagd/paramsd  │
│ 人脸校正  │       │ 侧向加速度    │              │ 参数学习       │
└──────────┘       └───────────────┘              └───────────────┘
    │                                                      │
    ▼                                                      ▼
┌──────────┐                                      ┌───────────────┐
│ UI 渲染   │                                      │ selfdrived    │
│ AR 叠加层 │                                      │ 告警生成       │
└──────────┘                                      └───────────────┘
```

## 7. 闭环反馈特性

calibrationd 与 modeld 形成了一个**闭环反馈系统**：

1. modeld 使用当前 `rpyCalib` 将原始图像变换到标准视角
2. supercombo model 在标准视角下估计自运动（cameraOdometry）
3. calibrationd 从自运动中估计新的 `rpyCalib`
4. 新的 `rpyCalib` 反馈给 modeld 更新图像变换
5. 循环迭代，标定逐步收敛

这个闭环的收敛速度取决于：
- **初始偏差大小**：偏差越小收敛越快
- **驾驶条件**：需要足够的高速直行段
- **Block 机制**：`INPUTS_NEEDED = 5` 个 block × `BLOCK_SIZE = 100` 样本 = 500 个有效样本才认为已校准

在 20 Hz 的 cameraOdometry 频率下，理论最快标定时间约为 25 秒（500 样本 / 20 Hz），但实际中由于条件门控的过滤，通常需要数分钟的高速直行。

## 8. 关键设计要点

1. **Roll 不估计**：`rpyCalib[0]` 始终为 0。openpilot 假设设备在横滚方向的安装是大致水平的，不对横滚做校正。

2. **闭环稳定性**：平滑过渡机制（`SMOOTH_CYCLES = 10`）和分块平均避免了标定值的突变，保护模型输入的连续性。

3. **散布监控**：当历史 block 的 pitch 散布 > 4° 或 yaw 散布 > 2° 时，系统推断设备可能被移动，自动进入 recalibrating 状态。

4. **安全边界**：pitch 和 yaw 都有硬限幅（`sanity_clip`），超出物理合理范围的值会被裁剪或标记为 invalid，防止极端标定值导致危险的控制行为。

5. **仿真适配**：仿真环境中视觉里程计可能低估速度（由于重复帧），calibrationd 会检测 `SIMULATION` 环境变量并放宽速度阈值。
