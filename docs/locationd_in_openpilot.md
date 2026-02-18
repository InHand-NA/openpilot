# openpilot locationd 模块技术分析

## 1. 概述

locationd 是 openpilot 的位姿估计与在线学习模块，由五个协作进程组成：

| 进程 | 功能 | 输出消息 |
|------|------|----------|
| **locationd** | IMU + 视觉里程计 EKF 融合，输出实时位姿 | `livePose` |
| **calibrationd** | 相机外参在线标定（pitch/yaw/height） | `liveCalibration` |
| **paramsd** | 车辆转向参数学习（转向比、刚度、角偏置） | `liveParameters` |
| **torqued** | 轮胎特性估计（摩擦系数、横向加速度因子） | `liveTorqueParameters` |
| **lagd** | 转向执行器延迟估计 | `liveDelay` |

所有进程仅在车辆行驶中运行（`only_onroad=True`），由 `system/manager/process_config.py` 管理。

## 2. 目录结构

```
selfdrive/locationd/
├── locationd.py              # 主进程：EKF 位姿估计
├── calibrationd.py           # 相机外参在线标定
├── paramsd.py                # 车辆参数学习
├── torqued.py                # 轮胎特性估计
├── lagd.py                   # 转向延迟估计
├── helpers.py                # 辅助工具（Pose、PoseCalibrator、PointBuckets）
├── models/
│   ├── pose_kf.py            # PoseKalman 定义（18 维状态 EKF）
│   ├── car_kf.py             # CarKalman 定义（9 维状态 EKF）
│   ├── constants.py          # ObservationKind 枚举
│   └── generated/            # sympy 生成的 C/Cython EKF 代码
└── test/                     # 单元测试
```

## 3. 数据流架构

```
                    ┌──────────────┐
                    │   camerad    │
                    │  (IMU 100Hz) │
                    └──────┬───────┘
                           │ accelerometer, gyroscope
                           ▼
┌──────────┐       ┌──────────────┐       ┌──────────────┐
│  modeld  │──────►│  locationd   │◄──────│  carState    │
│(20Hz cam │       │  (EKF 融合)  │       │  (车速等)    │
│ odometry)│       └──────┬───────┘       └──────────────┘
└────┬─────┘              │ livePose (20Hz)
     │                    ▼
     │             ┌──────────────┐
     │             │   paramsd    │──► liveParameters
     │             │  (车辆参数)  │
     │             └──────────────┘
     │
     │             ┌──────────────┐
     └────────────►│ calibrationd │──► liveCalibration
                   │ (相机标定)   │
                   └──────────────┘
                          │
    ┌─────────────────────┤
    ▼                     ▼
┌──────────┐       ┌──────────────┐
│ torqued  │       │    lagd      │
│(摩擦估计)│       │ (延迟估计)   │
└──────────┘       └──────────────┘
    │                     │
    ▼                     ▼
liveTorqueParameters   liveDelay
```

## 4. locationd — EKF 位姿估计

### 4.1 PoseKalman 状态向量（18 维）

```
状态索引    名称                 含义                    初始值    初始 P
[0:3]      NED_ORIENTATION      [roll, pitch, yaw]      0        0.01²
[3:6]      DEVICE_VELOCITY      [vx, vy, vz] m/s       0        10²
[6:9]      ANGULAR_VELOCITY     [ωx, ωy, ωz] rad/s     0        1²
[9:12]     GYRO_BIAS            陀螺仪偏置 rad/s        0        1²
[12:15]    ACCELERATION         [ax, ay, az] m/s²       0        100²
[15:18]    ACCEL_BIAS           加速度计偏置 m/s²       0        0.01²
```

### 4.2 动力学模型（预测步）

状态转移为简单的积分模型：

```
NED_ORIENTATION[t+1] = rot_to_euler(
    rot_from_euler(NED_ORIENTATION[t]) @ rot_from_euler(dt * angular_velocity[t])
)
DEVICE_VELOCITY[t+1] = DEVICE_VELOCITY[t] + dt * ACCELERATION[t]
其余状态（偏置等）保持不变（随机游走模型）
```

过程噪声 Q（对角阵）：

| 状态 | 噪声标准差 | 含义 |
|------|-----------|------|
| 姿态 | 0.001 rad | 缓慢变化 |
| 速度 | 0.01 m/s | 中等 |
| 角速度 | 0.1 rad/s | 较大 |
| 陀螺仪偏置 | 0.005/100 rad/s | 极缓慢漂移 |
| 加速度 | 3 m/s² | 较大（高动态） |
| 加速度计偏置 | 0.005 m/s² | 极缓慢漂移 |

### 4.3 观测模型（更新步）

**陀螺仪 (PHONE_GYRO, 100Hz)**

```
h(x) = angular_velocity + gyro_bias
R = diag([0.025², 0.025², 0.025²])
```

原始读数经坐标变换 `[-z, -y, -x]` 映射到设备坐标系。健全性检查：角速度模 < 10 rad/s。

**加速度计 (PHONE_ACCEL, 100Hz)**

```
h(x) = R_device_from_ned @ gravity + acceleration + ω × v + accel_bias
gravity = [0, 0, -9.81]  (NED 坐标系)
R = diag([0.5², 0.5², 0.5²])
```

加速度计观测包含重力分量（通过姿态旋转矩阵计算）和向心加速度（ω × v）。健全性检查：加速度模 < 100 m/s²。

**相机里程计 - 平移 (CAMERA_ODO_TRANSLATION, 20Hz)**

```
h(x) = velocity  (设备坐标系)
R = diag(trans_std²)  (动态调整，基础放大 2 倍)
```

**相机里程计 - 旋转 (CAMERA_ODO_ROTATION, 20Hz)**

```
h(x) = angular_velocity  (设备坐标系)
R = diag(rot_std²)  (动态调整，基础放大 10 倍)
```

相机里程计来自 modeld 的 `cameraOdometry` 消息，需经过标定变换（`device_from_calib`）后才送入 EKF。标准差放大（旋转 ×10、平移 ×2）是为了避免 EKF 过度信任视觉。

### 4.4 传感器校验与异常检测

| 检查项 | 条件 | 处理 |
|--------|------|------|
| 时间戳偏差 | \|sensor_time - log_time\| > 100ms | 丢弃该观测 |
| 滤波器回卷 | 观测时间 < 当前时间 - 0.8s | 丢弃 |
| 状态发散 | x 或 P 含 NaN/Inf | 重置 EKF |
| 坏输入累积 | 计数器 > 阈值 (2.0) | `inputsOK = False` |
| posenet 异常 | 近 20 帧 std 均值 / 前 20 帧 > 4 且 > 7 m/s | `posenetOK = False` |

坏输入计数器使用指数衰减恢复，恢复时间常数 10 秒。

### 4.5 输出：livePose

```
livePose:
  orientationNED:        [roll, pitch, yaw] ± std  (rad, NED 坐标系)
  velocityDevice:        [vx, vy, vz] ± std        (m/s, 设备坐标系)
  accelerationDevice:    [ax, ay, az] ± std        (m/s², 设备坐标系)
  angularVelocityDevice: [ωx, ωy, ωz] ± std       (rad/s, 设备坐标系)
  inputsOK:              所有传感器输入有效
  posenetOK:             视觉里程计有效
  sensorsOK:             IMU 有效
```

## 5. calibrationd — 相机外参在线标定

### 5.1 标定目标

估计相机（设备）相对于车辆标定坐标系的外参：
- `rpyCalib = [roll, pitch, yaw]`：设备相对标定坐标系的欧拉角
- `height`：相机离路面高度
- `wideFromDeviceEuler`：广角相机相对设备的欧拉角

### 5.2 标定输入条件

同时满足以下条件时才收集标定样本：
- 车速 > 15 m/s
- 视觉前向速度 trans[0] > min_speed
- 偏航角速度 |rot[2]| < 2 rad/s
- 姿态可靠性：arctan2(trans_std[1], trans[0]) < 0.25°
- 高度可靠性：road_transform_trans_std[2] < e^(-3.5)

### 5.3 标定算法

**观测计算**：从视觉平移反推姿态

```
observed_pitch = -arctan2(trans[2], trans[0])
observed_yaw   =  arctan2(trans[1], trans[0])
```

**Block 滑动窗口机制**：

- 每 100 个样本组成一个 block
- 维护 50 个 block 的环形缓冲区
- 需要 ≥ 5 个有效 block 才认为标定完成
- block 内使用线性衰减权重的移动平均

**状态机**：

```
uncalibrated ──(5+ blocks)──► calibrated ──(spread过大)──► recalibrating
                                  │                              │
                                  └──(RPY超范围)──► invalid      │
                                                                 │
                              calibrated ◄──(重新收敛)───────────┘
```

**稳定性监控**：当历史 block 中 pitch spread > 4° 或 yaw spread > 2° 时触发重标定。

### 5.4 持久化

标定结果存入 `Params["CalibrationParams"]`，每 10 个 block 周期写入一次，设备重启后自动恢复。

## 6. paramsd — 车辆参数学习

### 6.1 CarKalman 状态向量（9 维）

```
[0]   STIFFNESS           轮胎刚度因子          初始值 1.0
[1]   STEER_RATIO         转向比               初始值 CP.steerRatio
[2]   ANGLE_OFFSET        静态转向角偏置 (rad)  初始值 0
[3]   ANGLE_OFFSET_FAST   动态快速偏置 (rad)    初始值 0
[4:6] VELOCITY            [纵向, 横向] 速度     初始值 [10, 0]
[6]   YAW_RATE            偏航角速率            初始值 0
[7]   STEER_ANGLE         当前转向角            初始值 0
[8]   ROAD_ROLL           路面横坡角            初始值 0
```

### 6.2 车辆动力学模型

基于单轨模型（Bicycle Model），参考 Guiggiani 车辆动力学教科书：

```
dv/dt = A[0,0]*v + A[0,1]*r + B[0,0]*(δ - θ_offset) - g*sin(θ_road)
dr/dt = A[1,0]*v + A[1,1]*r + B[1,0]*(δ - θ_offset)
```

其中 A、B 矩阵由轮胎刚度（`stiffness × CP.tireStiffness`）、车辆质量、轴距、转向比等参数决定。

### 6.3 观测源

| 观测类型 | 来源 | 用途 |
|----------|------|------|
| STEER_ANGLE | carState | 方向盘转角 |
| ROAD_FRAME_YAW_RATE | livePose（经标定变换） | 偏航角速率 |
| ROAD_ROLL | livePose | 路面横坡角 |
| STIFFNESS / STEER_RATIO | 自身约束（高噪声） | 防止参数发散 |

### 6.4 学习条件与参数约束

- 仅在横向控制激活且车速 > 1 m/s 时学习
- 转向比范围：[0.5 × CP.steerRatio, 2.0 × CP.steerRatio]
- 刚度因子范围：[0.2, 5.0]
- 角偏置范围：±10°（滞后机制，有效边界 8°）

### 6.5 输出：liveParameters

```
steerRatio:            学习后的转向比
stiffnessFactor:       学习后的刚度因子
angleOffsetDeg:        静态角偏置（度）
angleOffsetAverageDeg: 平滑后的角偏置
roll:                  路面横坡角
```

## 7. torqued — 轮胎特性估计

### 7.1 估计目标

估计转向扭矩与横向加速度之间的关系参数：
- `latAccelFactor`：横向加速度缩放因子
- `frictionCoefficient`：轮胎-路面摩擦系数
- `latAccelOffset`：横向加速度偏置

### 7.2 Bucket 分组策略

按转向扭矩值分为 8 个区间：

```
[-0.5, -0.3], [-0.3, -0.2], [-0.2, -0.1], [-0.1, 0],
[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]
```

每个 bucket 有最小点数要求（边缘 100，中心 500），需要总点数 ≥ 4000 才输出有效估计。

### 7.3 数据收集条件

- 横向控制激活
- 无驾驶员转向干预
- 车速 > 15 m/s
- 横向加速度 ≤ 1 m/s²
- 至少 2 秒有效缓冲

### 7.4 拟合算法

使用 SVD（Total Least Squares）拟合转向扭矩 vs 横向加速度的线性关系：

```
steer_torque ≈ latAccelFactor × lat_accel + latAccelOffset + friction × sign(lat_accel)
```

参数经一阶低通滤波平滑输出，衰减常数范围 50-250。

## 8. lagd — 转向延迟估计

### 8.1 估计目标

估计转向指令（desired_curvature）到车辆实际响应（yaw_rate）之间的延迟时间。

### 8.2 算法：掩模归一化互相关（MNCC）

```
1. 收集 60 秒滑动窗口内的 desired vs actual 横向加速度
   - la_desired = desired_curvature × v_ego²
   - la_actual  = yaw_rate × v_ego

2. 使用 FFT 加速计算互相关函数

3. 搜索范围 0 ~ 1.0 秒，找到相关峰值

4. 抛物线插值精细化峰值位置（亚采样精度）
```

### 8.3 有效性判定

- Block 机制：100 采样/block，需要 5+ 有效 block
- 延迟标准差 < 0.1 秒
- 相关系数 > 0.9

### 8.4 输出：liveDelay

```
lateralDelay:             当前采用的延迟值（秒）
lateralDelayEstimate:     最新估计
lateralDelayEstimateStd:  估计标准差
status:                   unestimated / estimated / invalid
```

## 9. 坐标系定义

### 9.1 NED 坐标系（North-East-Down）

```
X: 正北  Y: 正东  Z: 向下
```

用于 `orientationNED`（绝对姿态基准）。旋转顺序：Roll(X) → Pitch(Y) → Yaw(Z)。

### 9.2 设备坐标系（Device Frame）

```
X: 前向  Y: 右向  Z: 向下
```

IMU 传感器输出经坐标变换后使用此坐标系。locationd 的速度、加速度、角速度输出均在此坐标系下。

### 9.3 标定坐标系（Calibrated Frame）

```
X: 前向  Y: 右向  Z: 向下（与路面平行）
```

calibrationd 提供 `device_from_calib` 变换（通过 `rpyCalib`），消除相机安装偏差。modeld 输出在标定坐标系下。

### 9.4 坐标变换

```python
# 标定 → 设备
device_from_calib = rot_from_euler(rpyCalib)
rot_device = device_from_calib @ rot_calib
trans_device = device_from_calib @ trans_calib

# NED ← 设备（由 EKF 状态中的 NED_ORIENTATION 决定）
ned_from_device = rot_from_euler(roll, pitch, yaw)
```

变换工具定义在 `common/transformations/orientation.py`。

## 10. EKF 实现框架（rednose）

### 10.1 代码生成流程

```
sympy 符号定义 (pose_kf.py)
    ↓ generate_code()
C 代码（状态转移 f、雅可比 F、观测 h、观测雅可比 H）
    ↓ Cython 编译
Python 可调用的 .so 文件 (models/generated/)
```

生成的核心函数：

```c
void pose_predict(double *x, double *P, double *Q, double dt);
void pose_update_4(double *x, double *P, double *z, double *R, double *ea);  // PHONE_GYRO
void pose_update_10(double *x, double *P, double *z, double *R, double *ea); // PHONE_ACCEL
void pose_update_13(double *x, double *P, double *z, double *R, double *ea); // CAM_TRANS
void pose_update_14(double *x, double *P, double *z, double *R, double *ea); // CAM_ROT
```

### 10.2 EKF 循环

```python
# KalmanFilter.predict_and_observe(t, kind, data, R)
# 内部执行：
# 1. 预测步
#    x⁻ = f(x, dt)
#    P⁻ = F · P · F^T + Q
# 2. 更新步（对每个观测）
#    y = z - h(x⁻)           # 观测残差
#    S = H · P⁻ · H^T + R    # 残差协方差
#    K = P⁻ · H^T · S⁻¹      # 卡尔曼增益
#    x = x⁻ + K · y
#    P = (I - K · H) · P⁻
```

支持时间回卷（rewind，最大 0.8 秒），用于处理乱序到达的传感器消息。

## 11. 参数持久化

各模块在 `Params` 中缓存状态，支持设备重启后快速恢复：

| 模块 | Param Key | 缓存内容 |
|------|-----------|----------|
| locationd | `LocationFilterInitialState` | EKF 状态 x 和 P |
| calibrationd | `CalibrationParams` | rpyCalib、height、validBlocks |
| paramsd | `LiveParametersV2` | steerRatio、stiffness、angleOffset |
| torqued | `LiveTorqueParameters` | friction、latAccelFactor、bucket 数据 |
| lagd | `LiveDelay` | lateralDelay、validBlocks |

恢复时会检查车型匹配（`CarParamsPrevRoute`），防止误用其他车辆的参数。

## 12. 与其他模块的交互

**上游（输入源）**：
- `camerad`：加速度计、陀螺仪原始数据（100Hz）
- `modeld`：相机里程计 `cameraOdometry`（20Hz）
- `pandad` → `carState`：车速、转向角等车身信息

**下游（消费者）**：
- `controlsd`：消费 `livePose`（姿态反馈）、`liveParameters`（转向参数）、`liveDelay`（延迟补偿）
- `modeld`：消费 `liveCalibration`（相机外参，用于图像空间 → 标定空间变换）
- `ui`：消费 `liveCalibration`（显示标定进度）

## 13. 关键设计决策

1. **双 EKF 架构**：PoseKalman 高频融合 IMU（实时控制），CarKalman 低频学习车辆参数（避免过拟合）。

2. **Block 平滑机制**：calibrationd 和 lagd 使用 block 累计 + 滑动窗口，减少瞬时噪声影响，需要多个 block 验证才认为有效。

3. **多层异常防护**：时间戳校验 → 范数检查 → 统计检验（std 尖峰）→ 坏输入计数器 → EKF 状态发散检测。

4. **视觉观测噪声放大**：旋转 ×10、平移 ×2，避免 EKF 过度信任视觉里程计（相比 IMU 更容易出现离群值）。

5. **符号代码生成**：通过 sympy → C → Cython 的流水线，将 EKF 的雅可比矩阵计算从 Python 转移到 C 级别，实现实时运行。

## 14. 关键常数

| 参数 | 值 | 含义 |
|------|-----|------|
| MAX_SENSOR_TIME_DIFF | 0.1s | 传感器时间戳与日志时间最大偏差 |
| MAX_FILTER_REWIND_TIME | 0.8s | EKF 允许回卷的最大时间 |
| ACCEL_SANITY_CHECK | 100 m/s² | 加速度上限 |
| ROTATION_SANITY_CHECK | 10 rad/s | 角速度上限 |
| INPUT_INVALID_LIMIT | 2.0 | 坏输入计数阈值 |
| INPUT_INVALID_RECOVERY | 10.0s | 坏输入恢复时间常数 |
| POSENET_STD_SPIKE_THRESHOLD | 4.0× | posenetOK 判定阈值 |
| calibrationd BLOCK_SIZE | 100 | 标定 block 样本数 |
| calibrationd INPUTS_NEEDED | 5 | 最少有效 block 数 |
| calibrationd MIN_SPEED_FILTER | 15 m/s | 标定最低车速 |
| paramsd STEER_RATIO 范围 | [0.5×, 2.0×] 标称值 | 转向比学习范围 |
| torqued MIN_POINTS_TOTAL | 4000 | 最少数据点 |
| lagd MAX_LAG | 1.0s | 延迟搜索上限 |
