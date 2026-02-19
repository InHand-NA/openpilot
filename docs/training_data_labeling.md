# openpilot 视觉模型训练数据标注方法

## 1. 概述

openpilot 的核心视觉模型 **supercombo** 采用**自监督学习**（self-supervised learning）为主的训练范式。与传统计算机视觉任务依赖人工标注不同，supercombo 的绝大部分训练标签来自车辆自身传感器（CAN 总线、IMU、GPS、雷达）和模型自身的历史预测，通过**行驶日志回放**自动生成。

### 1.1 核心原则

- **数据来源**：comma.ai 车队每天采集数百万英里的真实驾驶数据
- **标签生成**：非人工标注，而是从传感器融合结果和驾驶员行为中自动提取
- **Firehose Mode**：用户可选择上传更多训练数据，设备 UI 中的描述是：*"openpilot learns to drive by watching humans, like you, drive"*
- **验证手段**：通过 Process Replay 框架对比模型输出的一致性

### 1.2 模型架构概览

supercombo 由两个子网络组成：

| 子网络 | 输入 | 输出维度 | 职责 |
|--------|------|----------|------|
| **视觉网络** | 主相机 + 广角相机图像 (12ch × 128 × 256) | 1576 维 | 感知：车道线、路沿、前车、位姿、元事件 |
| **策略网络** | 视觉特征序列 (25×512) + desire 脉冲 + 交通习惯 | 1000 维 | 规划：未来 10 秒轨迹、驾驶意图 |

---

## 2. 数据采集与存储

### 2.1 日志记录系统

openpilot 的 `loggerd` 守护进程持续记录所有进程间消息和视频流：

```
每 60 秒为一个 segment，包含：
├── rlog.zst          # 全量 Cap'n Proto 消息日志（压缩）
├── qlog.zst          # 精简日志（约 1/10 大小）
├── fcamera.hevc      # 前向相机 H.265 视频 (1928×1208, 20fps)
├── ecamera.hevc      # 广角相机 H.265 视频
└── dcamera.hevc      # 驾驶员监控相机 H.265 视频
```

数据以 Route 为单位组织：`dongle_id|YYYY-MM-DD--HH-MM-SS`，每个 Route 包含多个连续 Segment。

### 2.2 关键传感器数据

日志中记录的与标签生成相关的传感器消息：

| 消息类型 | 来源 | 频率 | 提供信息 |
|----------|------|------|----------|
| `carState` | CAN 总线 | 100Hz | 车速、转向角、油门/制动踏板、转向灯 |
| `carControl` | 控制模块 | 100Hz | 期望曲率、期望加速度、横向控制激活状态 |
| `accelerometer` | IMU | 100Hz | 三轴加速度 |
| `gyroscope` | IMU | 100Hz | 三轴角速度 |
| `liveCalibration` | calibrationd | ~4Hz | 相机外参 (rpyCalib, height) |
| `livePose` | locationd | 20Hz | 融合后的位姿、速度、加速度 |
| `radarState` | radard | 20Hz | 雷达跟踪目标（距离、速度） |
| `liveParameters` | paramsd | ~4Hz | 转向比、刚度因子 |

### 2.3 Firehose Mode（数据众筹）

用户可以启用 Firehose Mode 上传更多驾驶数据用于训练：

- comma.ai 有选择性地拉取用户 segment 的子集
- 所有上游 openpilot 用户的数据均可用于训练
- 不需要特殊驾驶行为，正常驾驶即可
- 数据多样性来自 300+ 车型、30+ 国家的大规模车队

---

## 3. 模型输出结构与标签详解

### 3.1 输出分布类型

模型输出使用三种概率分布，对应不同的损失函数：

| 分布类型 | 解码方式 | 损失函数 | 适用输出 |
|----------|---------|----------|----------|
| **高斯 MDN** | mean + exp(log_std) | 负对数似然 (NLL) | 连续量：位置、速度、车道线坐标 |
| **Sigmoid** | 1/(1+exp(-x)) | 二元交叉熵 (BCE) | 概率：车道线存在、前车存在、元事件 |
| **Softmax** | exp(x)/sum(exp(x)) | 分类交叉熵 (CCE) | 分类分布：驾驶意图 |

### 3.2 采样网格

模型在非均匀采样网格上输出，近处密集远处稀疏：

```python
# 距离采样（车道线、路沿）：33 个点，0-192 米
X_IDXS[i] = 192.0 × (i/32)²
# → [0, 0.19, 0.75, 1.69, 3.0, ..., 150, 168, 192]

# 时间采样（轨迹规划）：33 个点，0-10 秒
T_IDXS[i] = 10.0 × (i/32)²
# → [0, 0.01, 0.04, 0.09, ..., 7.8, 8.8, 10.0]

# 前车时间采样：6 个点，均匀间隔
LEAD_T_IDXS = [0, 2, 4, 6, 8, 10]  # 秒
```

---

## 4. 各输出头的标签生成方法

### 4.1 轨迹规划（Plan）— 策略网络核心输出

**输出结构**：33 个时间点 × 15 维状态

```python
Plan.POSITION          [0:3]   # 未来位置 (x, y, z)  单位：米
Plan.VELOCITY          [3:6]   # 未来速度 (vx, vy, vz) 单位：m/s
Plan.ACCELERATION      [6:9]   # 未来加速度           单位：m/s²
Plan.T_FROM_CURRENT_EULER [9:12]  # 累积姿态变化 (roll, pitch, yaw) 单位：rad
Plan.ORIENTATION_RATE  [12:15] # 角速度               单位：rad/s
```

使用 MHP（Multi-Hypothesis Prediction）：5 个假设，选择最优 1 个。

**标签生成**：

| 维度 | 标签来源 | 方法 |
|------|---------|------|
| position | GPS + 车轮里程计 + IMU | 记录未来 10 秒的实际行驶轨迹，变换到当前帧的标定坐标系 |
| velocity | CAN 总线 vEgo + locationd | 未来各时刻的实际速度 |
| acceleration | CAN 总线 aEgo + IMU 融合 | 未来各时刻的实际加速度 |
| orientation | IMU 陀螺仪积分 | 未来各时刻相对当前的累积角度变化 |
| orientationRate | IMU 角速度 | 未来各时刻的实际角速度 |

**核心方法 — 未来轨迹回溯**：

这是 openpilot 自监督学习的关键。在训练时，模型在时刻 t 需要预测未来 10 秒的轨迹，而**标签就是实际发生的未来轨迹**——即时刻 t+1, t+2, ..., t+10s 的真实车辆状态。这些数据在回放日志中天然存在。

```
时间线：
t=0 (当前帧)  →  t=2s  →  t=4s  →  ...  →  t=10s
模型输入 ─┐    ├── 标签：实际位置 (from GPS/odometry)
          │    ├── 标签：实际速度 (from CAN vEgo)
          └──► ├── 标签：实际加速度 (from IMU)
               └── 标签：实际航向变化 (from gyroscope)
```

**MHP 假设选择**：5 个假设中，选择与实际轨迹均方误差最小的作为标签对应假设。

### 4.2 车道线（Lane Lines）— 视觉网络输出

**输出结构**：4 条线 × 33 个距离点 × 2 维 (y, z)

```
索引含义：
[0] 远左车道线 (far-left)     — 左相邻车道的左边界
[1] 近左车道线 (near-left)    — ego 车道的左边界
[2] 近右车道线 (near-right)   — ego 车道的右边界
[3] 远右车道线 (far-right)    — 右相邻车道的右边界
```

概率输出：`lane_lines_prob` 8 维 sigmoid，使用奇数索引 [1,3,5,7] 对应 4 条线。

**标签生成**：

车道线标签的生成是 openpilot 训练体系中最复杂的部分之一，采用多源融合策略：

**方法 1 — 高精地图 / HD Map**
- comma.ai 内部使用高精地图数据（如 OpenStreetMap 增强版）
- 提取车道边界的世界坐标，投影到标定坐标系
- 在 X_IDXS 距离点上插值得到 (y, z) 标签

**方法 2 — 离线视觉重建**
- 对行驶日志进行离线结构光恢复 (SfM)
- 从多帧图像中重建道路 3D 结构
- 提取车道标线的 3D 位置作为标签

**方法 3 — 前序模型的预测**（自蒸馏）
- 使用当前最佳模型对大量日志进行推理
- 将高置信度帧的预测结果作为下一代模型的伪标签
- 这是一种**自蒸馏**（self-distillation）策略

**方法 4 — 仿真环境精确标注**
- 使用 Carla 仿真器从地图 API 提取精确车道边界
- 实现在 `tools/dashcam/lane_ground_truth.py`
- 流程：Carla waypoint → 左右边界计算 → 世界坐标 → 标定坐标系 → X_IDXS 插值
- 主要用于**评估和验证**，也可作为训练数据的补充

### 4.3 路沿线（Road Edges）— 视觉网络输出

**输出结构**：2 条线 × 33 个距离点 × 2 维 (y, z)

```
[0] 左路沿    [1] 右路沿
```

**标签生成**：与车道线类似，主要来源：
- 高精地图中的道路边界数据
- 离线视觉重建
- 前序模型自蒸馏

### 4.4 前车目标（Leads）— 视觉网络输出

**输出结构**：3 个目标 × 6 个时间点 × 4 维 (x, y, v, a)

```
维度含义：
x — 前向距离 (m)
y — 横向偏移 (m)
v — 绝对速度 (m/s)   注意：非相对速度
a — 加速度 (m/s²)

3 个目标的 probTime 偏移：
lead[0]: 当前时刻 (t=0)
lead[1]: 2 秒后 (t=2)
lead[2]: 4 秒后 (t=4)
```

使用 MHP：2 个假设，选择最优 3 个目标。

**标签生成**：

| 维度 | 标签来源 | 方法 |
|------|---------|------|
| x (距离) | **雷达** | CAN 总线的雷达跟踪目标距离，毫米波雷达精度 ±0.1m |
| y (横向偏移) | 雷达 + 视觉 | 雷达提供角度，视觉提供精细横向定位 |
| v (速度) | **雷达** | 多普勒效应直接测量绝对径向速度 |
| a (加速度) | 雷达跟踪滤波 | 对速度时间序列求导，经卡尔曼滤波平滑 |

**雷达-视觉融合**（`radard` 模块）：
- 雷达提供精确的距离和速度（尤其在远距离），但角分辨率较低
- 视觉提供精确的角度和横向位置，但距离估计依赖几何推算
- 融合结果作为前车标签：距离信赖雷达，横向信赖视觉

**时间序列标签**：
- 前车 6 个时间点的状态同样使用"未来回溯"方法
- 即记录 t=0, 2, 4, 6, 8, 10 秒后前车的实际位置和速度

### 4.5 相机位姿（Pose）— 视觉网络输出

**输出结构**：6 维 (tx, ty, tz, roll, pitch, yaw) + 6 维标准差

```
平移：tx, ty, tz (m/s) — 相机速度
旋转：roll, pitch, yaw (rad/s) — 相机角速度
```

发布为 `cameraOdometry` 消息，下游由 locationd EKF 融合。

**标签生成**：

| 维度 | 标签来源 |
|------|---------|
| 平移速度 | CAN 总线车速 (vEgo) + IMU 加速度积分 |
| 旋转角速度 | IMU 陀螺仪（经偏置校正） |
| 高精标签 | locationd 的 EKF 融合输出（livePose） |

实际上，pose 输出与 locationd 形成**双向反馈**：
1. 模型输出 pose → locationd 作为视觉观测输入 EKF
2. locationd 的融合输出 → 回放时作为 pose 训练标签

### 4.6 广角相机欧拉角（Wide From Device Euler）

**输出结构**：3 维 (roll, pitch, yaw) + 3 维标准差

**标签生成**：
- 通过离线标定或在线 calibrationd 获得广角相机相对设备的固定外参
- 每台设备这组参数基本恒定，标签是制造标定值或长期在线估计值

### 4.7 路面变换（Road Transform）

**输出结构**：6 维 + 6 维标准差

其中 `roadTransformTrans[2]` 用于估计路面高度。

**标签生成**：
- 来自 calibrationd 的路面几何估计
- 通过多帧视觉特征的高度一致性约束

### 4.8 元事件预测（Meta）— 视觉网络输出

**输出结构**：55 维 sigmoid，覆盖未来 2-10 秒的事件预测

```python
Meta.ENGAGED           [0:1]       # 驾驶员参与度
Meta.GAS_DISENGAGE     [1:31:6]    # 油门导致脱离 (5 个时间点)
Meta.BRAKE_DISENGAGE   [2:31:6]    # 制动导致脱离
Meta.STEER_OVERRIDE    [3:31:6]    # 方向盘接管
Meta.HARD_BRAKE_3      [4:31:6]    # 急刹 ≥3 m/s²
Meta.HARD_BRAKE_4      [5:31:6]    # 急刹 ≥4 m/s²
Meta.HARD_BRAKE_5      [6:31:6]    # 急刹 ≥5 m/s²
Meta.GAS_PRESS         [31:55:4]   # 油门踩下 (6 个时间点)
Meta.BRAKE_PRESS       [32:55:4]   # 制动踩下
Meta.LEFT_BLINKER      [33:55:4]   # 左转灯亮
Meta.RIGHT_BLINKER     [34:55:4]   # 右转灯亮
```

**标签生成 — 全部来自 CAN 总线原始信号**：

| 事件 | CAN 信号来源 | 标签规则 |
|------|-------------|---------|
| 油门脱离 | `carState.gasPressed` | 驾驶员踩油门踏板 → 1.0 |
| 制动脱离 | `carState.brakePressed` | 驾驶员踩制动踏板 → 1.0 |
| 方向盘接管 | `carState.steeringPressed` + 角度变化 | 检测到方向盘扭矩 → 1.0 |
| 急刹 3/4/5 | `carState.aEgo` | aEgo < -3/-4/-5 m/s² → 1.0 |
| 油门/制动踩下 | CAN 踏板信号 | 直接读取踏板状态 |
| 转向灯 | `carState.leftBlinker / rightBlinker` | 直接读取灯光状态 |

**时间标签**：每个事件在 META_T_IDXS = [2, 4, 6, 8, 10] 秒的时间窗口内是否发生。训练时使用"未来回溯"：查看未来 2/4/6/8/10 秒内该事件是否实际发生。

**FCW（前碰撞预警）触发逻辑**：
```
hardBrakePredicted = (最近 5 帧 hardBrake5 > [0.05,0.05,0.15,0.15,0.15]).all()
                   AND (最近 2 帧 hardBrake3 > [0.7, 0.7]).all()
```

### 4.9 驾驶意图（Desire）— 双网络输出

**视觉网络**输出 `desire_pred`：未来 4 个时段 × 8 类 softmax
**策略网络**输出 `desire_state`：当前 8 类 softmax

```
8 类驾驶意图：
[0] none           — 无意图（直行）
[1] turnLeft       — 左转
[2] turnRight      — 右转
[3] laneChangeLeft — 左变道
[4] laneChangeRight— 右变道
[5] keepLeft       — 靠左行驶
[6] keepRight      — 靠右行驶
[7] (保留)
```

**标签生成 — DesireHelper 状态机**：

标签来自 `selfdrive/controls/lib/desire_helper.py` 的状态机逻辑：

```
触发条件（同时满足）：
1. 检测到单侧转向灯 (carState.leftBlinker XOR rightBlinker)
2. 车速 > 20 mph (~8.9 m/s)
3. 驾驶员主动施加与转向灯方向一致的方向盘扭矩
4. 对应方向无盲点障碍物

状态机：
off → preLaneChange → laneChangeStarting → laneChangeFinishing → off

desire 映射：
- off / preLaneChange → none
- laneChangeStarting / Finishing + left → laneChangeLeft
- laneChangeStarting / Finishing + right → laneChangeRight
```

**训练时标签**：desire 在变道过程中为 one-hot 脉冲，非变道时为 `none`。策略网络的输入 `desire_pulse` 是上升沿脉冲（25 帧 × 8 维），只在状态转换时短暂激活。

### 4.10 驾驶置信度（Confidence）

**非模型直接输出**，而是从 meta 事件概率后处理计算：

```python
# 融合三种脱离概率
any_disengage = 1 - (1-brake)×(1-gas)×(1-steer)

# 5 帧滑动缓冲取对角线平均
score = mean(diag(buffer))

# 分级
green:  score < 0.01165   (低脱离风险)
yellow: score < 0.06157   (中等风险)
red:    score ≥ 0.06157   (高风险)
```

---

## 5. 训练数据管道

### 5.1 端到端流程

```
┌─────────────────────────────────────────────────────────┐
│ 阶段 1：数据采集（设备端）                                │
│                                                         │
│ 真实驾驶 → loggerd 记录 → rlog.zst + 视频               │
│                    ↓                                     │
│ uploader → Comma 云端存储                                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段 2：标签提取（离线）                                  │
│                                                         │
│ 读取 rlog → 提取 CAN/IMU/雷达/GPS 信号                   │
│              ↓                                           │
│ 对每帧图像，从未来的日志中回溯提取：                        │
│   • 未来 10 秒的车辆轨迹 → plan 标签                     │
│   • 未来事件（踏板/转向灯） → meta 标签                   │
│   • 雷达跟踪结果 → lead 标签                              │
│   • 地图/视觉重建/自蒸馏 → lane/edge 标签                 │
│   • IMU+GPS 融合 → pose 标签                             │
│   • DesireHelper 状态机 → desire 标签                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段 3：模型训练                                         │
│                                                         │
│ 图像 (YUV420) + 标签 → 多任务端到端训练                   │
│   • 视觉网络：感知任务 (MDN/BCE/CCE 损失)                │
│   • 策略网络：规划任务 (MHP-MDN 损失)                    │
│   • 联合优化，梯度同时更新两个网络                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段 4：验证                                             │
│                                                         │
│ Process Replay：在固定日志上对比新旧模型输出               │
│ Carla 仿真：精确 GT 定量评估                              │
│ 车队部署：A/B 测试                                       │
└─────────────────────────────────────────────────────────┘
```

### 5.2 未来回溯法（核心标注策略）

openpilot 标注体系的核心思想是**利用未来已知的事实作为当前预测的标签**：

```
日志时间线：
... t-2s  t-1s  [t=当前帧]  t+1s  t+2s  ...  t+10s ...

模型在 t 时刻的输入：t 和 t-1 的相机图像
模型在 t 时刻的预测目标：t ~ t+10s 的轨迹和事件

标签提取：从日志中读取 t+1s ~ t+10s 实际发生的：
  - 车辆位置、速度、加速度 (from CAN/IMU/GPS)
  - 驾驶员操作（踏板、方向盘、转向灯）(from CAN)
  - 前车状态变化 (from 雷达跟踪)
```

这种方法的优势：
- **零标注成本**：不需要人工标注
- **标签精度高**：传感器融合结果的精度远高于人工标注
- **数据量无上限**：每次驾驶都自动产生训练数据
- **分布一致**：训练数据与部署场景完全一致

### 5.3 自蒸馏（模型迭代策略）

车道线和路沿等难以从传感器直接获取的标签，使用自蒸馏方法：

```
模型 V(n) ──推理──► 对海量日志预测 ──筛选高置信度帧──► 伪标签
                                                        ↓
模型 V(n+1) ◄──训练──── 图像 + 伪标签 + 传感器标签 ◄────┘
```

每代模型在上一代的基础上改进，形成正向迭代循环。

---

## 6. 损失函数设计

### 6.1 各输出头的损失权重（推断）

| 输出头 | 损失函数 | 相对权重 | 说明 |
|--------|---------|---------|------|
| plan (position) | MHP-MDN NLL | **高** | 轨迹预测是核心任务 |
| plan (velocity) | MHP-MDN NLL | **高** | 速度直接影响纵向控制 |
| plan (acceleration) | MHP-MDN NLL | 中 | 平滑约束 |
| plan (orientation) | MHP-MDN NLL | **高** | 航向角用于曲率计算 |
| plan (orientationRate) | MHP-MDN NLL | 中 | 角速度平滑约束 |
| lane_lines | MDN NLL | 中 | 车道线几何 |
| lane_lines_prob | BCE | 低 | 车道线存在概率 |
| road_edges | MDN NLL | 中 | 路沿几何 |
| lead (x, y, v, a) | MHP-MDN NLL | 中 | 前车状态预测 |
| lead_prob | BCE | 低 | 前车存在概率 |
| pose | MDN NLL | 低 | 相机里程计（辅助任务） |
| meta 事件 | BCE | 中 | 脱离/急刹预测 |
| desire_state | CCE | 中 | 当前驾驶意图分类 |
| desire_pred | CCE | 中 | 未来意图预测 |

### 6.2 MDN 损失详解

对于高斯 MDN 输出（mean μ, std σ），负对数似然为：

```
L = 0.5 × log(2π) + log(σ) + (y - μ)² / (2σ²)
```

模型网络输出 log_std，通过 `exp(clip(log_std, -inf, 11))` 得到 σ。

### 6.3 MHP 损失详解

对于 5 个假设的轨迹预测：
1. 计算每个假设与真实轨迹的 MDN NLL
2. 选择 NLL 最小的假设作为"胜出"假设
3. 仅对胜出假设计算梯度
4. 同时对假设权重 softmax 计算分类损失

---

## 7. 验证与评估

### 7.1 Process Replay

```python
# 在固定日志上重放模型
lr = LogReader(route_url)
frs = {camera: FrameReader(url) for camera in cameras}
output_msgs = replay_process(modeld_config, lr, frs)

# 对比新旧模型输出
results = compare_logs(reference_msgs, output_msgs, tolerance=0.3)
```

确保模型更新后输出在可接受范围内变化。

### 7.2 Carla 仿真评估

使用 `tools/dashcam/` 中的评估工具：

```bash
python tools/dashcam/run.py --perfect-cam --eval-lanes --max-frames 1000
```

定量指标（来自 `lane_evaluator.py`）：
- **横向误差**：MAE/RMSE，分 near (0-30m) / mid (30-60m) / far (60-100m)
- **检测率**：prob-based 和 position-based (>80% 点在 0.5m 内)
- **逐点精度率**：0.3m / 0.5m / 1.0m 阈值下的达标率
- **车道宽度误差**：左右线宽度差的 MAE

### 7.3 车队 A/B 测试

新模型部署到部分车队，对比：
- 脱离率（disengagement rate）
- 驾驶员干预频率
- FCW 误触发率
- 置信度分布

---

## 8. 与传统标注方法的对比

| 维度 | 传统方法（如 nuScenes） | openpilot 自监督方法 |
|------|----------------------|---------------------|
| 标注方式 | 人工标注 + 半自动工具 | 传感器融合 + 未来回溯 |
| 标注成本 | 高（$10-100/帧） | 几乎为零 |
| 数据规模 | 十万~百万帧 | 数十亿帧 |
| 标签类型 | 2D/3D 框、语义分割 | 连续轨迹、概率分布 |
| 域适配 | 可能存在域偏移 | 训练=部署场景 |
| 更新频率 | 月/年级别 | 每日持续积累 |
| 车道线标签 | 人工标注多段线 | 地图 + 视觉重建 + 自蒸馏 |
| 前车标签 | 3D 框标注 | 雷达融合（距离/速度直接测量） |

---

## 9. 关键源码文件

| 文件 | 功能 |
|------|------|
| `selfdrive/modeld/modeld.py` | 模型推理主循环 |
| `selfdrive/modeld/parse_model_outputs.py` | 输出解析（MDN/sigmoid/softmax） |
| `selfdrive/modeld/fill_model_msg.py` | 输出 → cereal 消息转换 |
| `selfdrive/modeld/constants.py` | 常量定义（采样网格、输出维度） |
| `selfdrive/controls/lib/desire_helper.py` | 驾驶意图标签状态机 |
| `selfdrive/locationd/locationd.py` | IMU+视觉 EKF 融合（pose 标签源） |
| `selfdrive/locationd/calibrationd.py` | 相机标定（road_transform 标签源） |
| `selfdrive/radard/radard.py` | 雷达-视觉融合（lead 标签源） |
| `system/loggerd/loggerd.cc` | 日志记录守护进程 |
| `tools/lib/logreader.py` | 日志读取工具 |
| `tools/dashcam/lane_ground_truth.py` | Carla 车道线 GT 提取 |
| `tools/dashcam/lane_evaluator.py` | 车道线评估指标 |
| `selfdrive/test/process_replay/model_replay.py` | 模型回放验证 |
| `selfdrive/ui/layouts/settings/firehose.py` | Firehose Mode UI |

---

## 10. 总结

openpilot 的训练数据标注体系是一个**高度自动化的闭环系统**：

1. **采集**：车队自然驾驶产生海量多模态数据
2. **标注**：传感器融合 + 未来回溯 + 自蒸馏，无需人工标注
3. **训练**：多任务端到端学习，联合优化感知与规划
4. **验证**：Process Replay 保证一致性，仿真评估保证精度
5. **部署**：更好的模型产生更好的驾驶 → 更好的训练数据 → 正向循环

这种"通过观察人类驾驶来学习驾驶"的方法，使 openpilot 能够以极低成本持续改进模型，是其核心竞争力之一。
