# IMU 传感器数据在 openpilot 系统中的作用

## 1. 概述

IMU（惯性测量单元）是 openpilot 系统中最关键的传感器之一。它提供高频率（104 Hz）的加速度和角速率测量，与摄像头视觉里程计（20 Hz）互补融合，为整个系统提供实时的车辆姿态、速度和加速度估计。

IMU 数据贯穿 openpilot 的多个核心子系统：

| 子系统 | 作用 |
|--------|------|
| **locationd** | 扩展卡尔曼滤波器的核心观测源，估计车辆姿态和运动状态 |
| **calibrationd** | 通过偏航角速率门控标定条件 |
| **controlsd** | 将滤波后的姿态和角速率下发给车辆控制器 |
| **torqued** | 利用角速率计算侧向加速度，标定横向扭矩模型 |
| **selfdrived** | 监控 IMU 数据健康状态，触发传感器故障告警 |

## 2. 硬件与数据采集

### 2.1 传感器硬件

openpilot 运行在 comma 3X 硬件上，使用 **LSM6DS3** 六轴 IMU（集成三轴加速度计 + 三轴陀螺仪），通过 I2C 总线（地址 `0x6A`）通信。

| 参数 | 加速度计 | 陀螺仪 |
|------|----------|--------|
| 采样率 | 104 Hz | 104 Hz |
| 量程 | +/-2g | +/-250 dps |
| 输出寄存器 | `0x28` (OUTX_L_XL) | `0x22` (OUTX_L_G) |
| 数据格式 | 16-bit 有符号整数 x 3轴 | 16-bit 有符号整数 x 3轴 |

### 2.2 数据采集流程（sensord）

IMU 数据由 `system/sensord/sensord.py` 采集并发布，采用 **GPIO 中断驱动** 方式以保证时间精度：

```
GPIO Pin 84 中断 → 读取硬件时间戳 → I2C 读取寄存器 → 坐标变换 → 发布消息
```

**加速度计**（`system/sensord/sensors/lsm6ds3_accel.py`）：

```python
scale = 9.81 * 2.0 / (1 << 15)      # 原始值 → m/s^2
b = self.read(0x28, 6)               # 读 6 字节（3 轴 x 16-bit）
x = self.parse_16bit(b[0], b[1]) * scale
y = self.parse_16bit(b[2], b[3]) * scale
z = self.parse_16bit(b[4], b[5]) * scale
a.v = [y, -x, z]                     # 芯片坐标 → 设备坐标
```

**陀螺仪**（`system/sensord/sensors/lsm6ds3_gyro.py`）：

```python
scale = (8.75 / 1000.0) * (pi / 180.0)  # mdps → rad/s
xyz = [y * scale, -x * scale, z * scale] # 芯片坐标 → 设备坐标
```

两个传感器在硬件层面共享数据就绪（Data Ready）中断信号，确保加速度和角速率的时间戳严格对齐。

### 2.3 消息格式（cereal）

IMU 数据通过 cereal 消息队列发布为独立的 `accelerometer` 和 `gyroscope` 服务（`cereal/log.capnp`）：

```capnp
struct SensorEventData {
  timestamp @3 :Int64;                    # 纳秒级硬件时间戳
  union {
    acceleration @4 :SensorVec;           # 加速度（m/s^2）
    gyroUncalibrated @12 :SensorVec;      # 未校准角速率（rad/s）
  }
  source @8 :SensorSource;               # 传感器来源标识
}
```

服务定义（`cereal/services.py`）：

```python
"accelerometer": (True, 104., 104)    # 日志记录, 104 Hz, qlog 采样率
"gyroscope":     (True, 104., 104)
```

## 3. 核心处理：locationd 扩展卡尔曼滤波器

### 3.1 总体架构

locationd（`selfdrive/locationd/locationd.py`）是 IMU 数据的主要消费者。它运行一个 **18 维扩展卡尔曼滤波器（EKF）**，融合 IMU 和摄像头里程计来估计车辆的完整运动状态。

```
加速度计 (104 Hz) ──┐
                    ├──→ EKF (predict_and_observe) ──→ livePose (20 Hz)
陀螺仪 (104 Hz) ───┤
                    │
摄像头里程计 (20 Hz) ┘
```

### 3.2 状态向量（18 维）

卡尔曼滤波器的状态定义在 `selfdrive/locationd/models/pose_kf.py`：

| 状态分量 | 维度 | 含义 | 单位 |
|----------|------|------|------|
| `NED_ORIENTATION` | 3 | 横滚、俯仰、偏航 | rad |
| `DEVICE_VELOCITY` | 3 | 设备坐标系速度 | m/s |
| `ANGULAR_VELOCITY` | 3 | 角速率 | rad/s |
| `GYRO_BIAS` | 3 | 陀螺仪偏置 | rad/s |
| `ACCELERATION` | 3 | 加速度 | m/s^2 |
| `ACCEL_BIAS` | 3 | 加速度计偏置 | m/s^2 |

其中 **GYRO_BIAS** 和 **ACCEL_BIAS** 是 EKF 在线估计的传感器偏置，这是 IMU 航位推算的关键——偏置漂移是纯惯性导航的主要误差来源，EKF 通过视觉里程计的约束持续修正这些偏置。

### 3.3 观测方程

**陀螺仪观测模型**：

```
z_gyro = omega + bias_gyro
```

直接将角速率和偏置相加，偏置由 EKF 在线估计。

**加速度计观测模型**：

```
z_accel = R(NED→Device) · g + a + omega x v + bias_accel
```

其中各项含义：
- `R(NED→Device) · g`：重力在设备坐标系中的投影（g = 9.81 m/s^2）
- `a`：设备坐标系中的线性加速度
- `omega x v`：向心加速度（角速率与速度的叉积）
- `bias_accel`：加速度计偏置

加速度计观测模型较复杂，因为加速度计测量的是**比力**（specific force），包含重力分量。EKF 需要利用当前姿态估计来分离重力和线性加速度。

### 3.4 观测噪声

```python
obs_noise = {
  PHONE_GYRO:              diag([0.025^2, 0.025^2, 0.025^2]),     # 25 mrad/s
  PHONE_ACCEL:             diag([0.5^2,   0.5^2,   0.5^2]),       # 0.5 m/s^2
  CAMERA_ODO_TRANSLATION:  diag([0.5^2,   0.5^2,   0.5^2]),       # 0.5 m/s
  CAMERA_ODO_ROTATION:     diag([0.05^2,  0.05^2,  0.05^2]),      # 50 mrad/s
}
```

陀螺仪的噪声（25 mrad/s）远低于摄像头旋转估计（50 mrad/s），说明在旋转估计方面 **IMU 的精度高于视觉**。而在平移估计方面，两者噪声水平相当。

### 3.5 数据预处理

在进入 EKF 之前，IMU 数据经过多重校验（`locationd.py:99-143`）：

**1) 坐标变换**：

```python
v = msg.acceleration.v
meas = np.array([-v[2], -v[1], -v[0]])   # 设备坐标 → NED 相关坐标系
```

sensord 发布的数据已从芯片坐标映射到设备坐标（`[y, -x, z]`），locationd 进一步做 `[-z, -y, -x]` 变换，将数据映射到 EKF 使用的坐标系。

**2) 时间戳校验**：

```python
MAX_SENSOR_TIME_DIFF = 0.1  # 100 ms
# 传感器时间戳与日志时间偏差超过 100ms 则丢弃
```

**3) 传感器源过滤**：

```python
# 某些硬件有双 IMU（bmx055 + lsm6ds3），忽略 bmx055 避免重复
def _validate_sensor_source(self, source):
    return source != log.SensorEventData.SensorSource.bmx055
```

**4) 幅值检查**：

```python
ACCEL_SANITY_CHECK = 100.0     # m/s^2（约 10g）
ROTATION_SANITY_CHECK = 10.0   # rad/s（约 573 deg/s）
```

**5) 陀螺仪与视觉交叉校验**：

```python
# 陀螺偏航角速率 vs 摄像头里程计偏航角速率
gyro_bias = self.kf.x[States.GYRO_BIAS]
gyro_camodo_yawrate_err = abs((meas[2] - gyro_bias[2]) - camodo_yawrate)
gyro_valid = gyro_camodo_yawrate_err < 30 * camodo_yawrate_std
```

这一步确保陀螺仪数据与摄像头里程计估计的偏航角速率一致，能够检测陀螺仪硬件故障。

## 4. 下游消费者

### 4.1 controlsd — 车辆控制

`selfdrive/controls/controlsd.py` 订阅 EKF 的输出 `livePose`，经标定变换后下发给车辆控制器：

```python
device_pose = Pose.from_live_pose(sm['livePose'])
calibrated_pose = pose_calibrator.build_calibrated_pose(device_pose)

CC.orientationNED = calibrated_pose.orientation.xyz.tolist()     # [roll, pitch, yaw]
CC.angularVelocity = calibrated_pose.angular_velocity.xyz.tolist()  # [d_roll, d_pitch, d_yaw]
```

`orientationNED` 中的 **roll** 角用于补偿重力在横向的分量（坡道上车辆会受到侧向重力分力）。`angularVelocity` 中的 **yaw rate** 是横向控制的核心反馈信号。

### 4.2 torqued — 横向扭矩标定

`selfdrive/locationd/torqued.py` 利用 EKF 输出的角速率和姿态计算侧向加速度，用于在线标定横向控制的扭矩模型：

```python
yaw_rate = calibrated_pose.angular_velocity.yaw
roll = device_pose.orientation.roll
lateral_acc = (v_ego * yaw_rate) - (sin(roll) * 9.81)
```

公式含义：
- `v_ego * yaw_rate`：运动学侧向加速度
- `sin(roll) * g`：坡道引起的侧向重力分力补偿

通过积累 `(方向盘扭矩, 侧向加速度)` 数据对，torqued 在线拟合扭矩-侧向加速度的线性关系，持续优化横向控制的精度。

### 4.3 calibrationd — 相机标定

`selfdrive/locationd/calibrationd.py` 不直接使用原始 IMU 数据，但通过以下条件间接依赖：

```python
straight_and_fast = ((v_ego > MIN_SPEED_FILTER) and
                     (trans[0] > min_speed) and
                     (abs(rot[2]) < MAX_YAW_RATE_FILTER))   # 偏航角速率 < 2 deg/s
```

`rot[2]`（偏航角速率）来源于摄像头里程计，而 locationd 中 IMU 的陀螺仪与摄像头里程计持续交叉校验，确保这个值的可靠性。标定只在**高速直行**时更新，避免弯道和低速时的噪声污染标定结果。

### 4.4 selfdrived — 传感器健康监控

`selfdrive/selfdrived/selfdrived.py` 订阅 `accelerometer` 和 `gyroscope` 消息，监控数据新鲜度：

```python
sensor_packets = ["accelerometer", "gyroscope"]

# 如果超过 10 秒未收到传感器数据，触发告警
if any((frame - recv_frame[s]) * DT_CTRL > 10. for s in sensor_packets):
    events.add(EventName.sensorDataInvalid)
```

`sensorDataInvalid` 事件会导致 openpilot 禁用辅助驾驶功能，确保在 IMU 故障时的安全降级。

## 5. 完整数据流

```
┌─────────────────────────────────────────────────────────┐
│ 硬件层                                                   │
│                                                         │
│  LSM6DS3 IMU (I2C @ 0x6A)                               │
│  ├─ 加速度计：+/-2g, 104 Hz                               │
│  └─ 陀螺仪：+/-250 dps, 104 Hz                           │
│        │                                                │
│  GPIO Pin 84 中断触发                                     │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ sensord (system/sensord/sensord.py)                      │
│                                                         │
│  ├─ LSM6DS3_Accel.get_event()                           │
│  │   原始 16-bit → m/s^2, 坐标变换 [y, -x, z]            │
│  │   → 发布 "accelerometer" 消息 @ 104 Hz                │
│  │                                                      │
│  └─ LSM6DS3_Gyro.get_event()                            │
│      原始 16-bit → rad/s, 坐标变换 [y, -x, z]             │
│      → 发布 "gyroscope" 消息 @ 104 Hz                    │
└────────┬────────────────────────────────────────────────┘
         │ cereal 消息队列
         ▼
┌─────────────────────────────────────────────────────────┐
│ locationd (selfdrive/locationd/locationd.py)             │
│                                                         │
│  数据预处理：                                              │
│  ├─ 时间戳校验（< 100 ms 偏差）                            │
│  ├─ 传感器源过滤（排除 bmx055）                             │
│  ├─ 坐标变换 [-v[2], -v[1], -v[0]]                       │
│  ├─ 幅值检查（加速度 < 100 m/s^2, 角速率 < 10 rad/s）      │
│  └─ 陀螺仪-视觉交叉校验                                    │
│                                                         │
│  扩展卡尔曼滤波器（18 维状态）：                              │
│  ├─ 姿态 [roll, pitch, yaw]                              │
│  ├─ 速度 [vx, vy, vz]                                   │
│  ├─ 角速率 [wx, wy, wz]                                  │
│  ├─ 陀螺偏置 [bg_x, bg_y, bg_z]      ← 在线估计           │
│  ├─ 加速度 [ax, ay, az]                                  │
│  └─ 加速度偏置 [ba_x, ba_y, ba_z]    ← 在线估计           │
│                                                         │
│  → 发布 "livePose" 消息 @ 20 Hz                          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ 下游消费者                                                │
│                                                         │
│  controlsd ─→ 姿态补偿 + 角速率反馈 → carcontroller       │
│  torqued   ─→ 侧向加速度 = v·yaw_rate - g·sin(roll)     │
│  selfdrived ─→ 传感器健康监控 → sensorDataInvalid 告警     │
│  calibrationd ─→ 偏航角速率门控标定条件                     │
└─────────────────────────────────────────────────────────┘
```

## 6. IMU 故障场景与容错

openpilot 针对 IMU 故障有完整的测试覆盖（`selfdrive/locationd/test/test_locationd_scenarios.py`）：

| 故障场景 | 测试名 | 系统行为 |
|----------|--------|----------|
| 陀螺仪完全丢失 | `GYRO_OFF` | 仅依赖视觉里程计，精度下降 |
| 陀螺仪单次尖峰 | `GYRO_SPIKE_MIDWAY` | 幅值检查丢弃异常值 |
| 陀螺仪持续尖峰 | `GYRO_CONSISTENT_SPIKES` | 交叉校验 + 幅值检查拒绝 |
| 加速度计完全丢失 | `ACCEL_OFF` | 姿态估计退化，速度仅依赖视觉 |
| 加速度计单次尖峰 | `ACCEL_SPIKE_MIDWAY` | 幅值检查丢弃异常值 |
| 加速度计持续尖峰 | `ACCEL_CONSISTENT_SPIKES` | 幅值检查持续拒绝 |
| 时间戳异常 | `SENSOR_TIMING_SPIKE_MIDWAY` | 时间戳校验丢弃该帧 |

## 7. 仿真环境中的 IMU

在 Carla/MetaDrive 仿真中，IMU 数据由 `tools/sim/lib/simulated_sensors.py` 生成：

```python
def send_imu_message(self, simulator_state):
    # 加速度计
    dat.accelerometer.acceleration.v = [ax, ay, az]
    self.pm.send('accelerometer', dat)

    # 陀螺仪
    dat.gyroscope.gyroUncalibrated.v = [gx, gy, gz]
    self.pm.send('gyroscope', dat)
```

仿真环境直接从模拟器获取理想的加速度和角速率值，不经过硬件驱动，因此没有真实传感器的噪声和偏置特性。这意味着仿真中 EKF 的偏置估计会趋近于零。

## 8. 关键设计要点总结

1. **高频 IMU + 低频视觉 = 互补融合**：IMU 提供 104 Hz 的高频运动更新，视觉里程计提供 20 Hz 的绝对约束，EKF 将两者融合得到最优估计。

2. **在线偏置估计**：EKF 持续估计 6 个偏置参数（3 个陀螺偏置 + 3 个加速度偏置），补偿传感器漂移。

3. **多层容错**：时间戳校验 → 传感器源过滤 → 幅值检查 → 视觉交叉校验 → EKF 协方差监控，确保异常数据不会污染状态估计。

4. **重力分离**：加速度计观测模型通过姿态估计分离重力分量，提取线性加速度，同时 roll 角用于补偿坡道对横向控制的影响。

5. **安全降级**：IMU 数据丢失超过 10 秒触发 `sensorDataInvalid`，系统禁用辅助驾驶。
