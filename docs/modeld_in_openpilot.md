# modeld 模块技术文档

## 1. 概述

modeld 是 openpilot 的核心感知与决策模块，负责将摄像头图像转换为驾驶规划与控制指令。它运行一个端到端的深度神经网络（称为 supercombo 模型），以 20Hz 频率处理双路摄像头输入，输出包括行驶轨迹规划、车道线检测、前车检测、相机里程计等关键信息。

modeld 的核心设计特点是**视觉-策略分离架构**：视觉网络（Vision Network）负责从图像中提取特征和感知信息，策略网络（Policy Network）基于视觉特征的时间序列生成行驶规划。这种设计使得策略网络可以利用多帧历史特征进行时序推理。

**源码入口**: `selfdrive/modeld/modeld.py`

## 2. 系统架构与数据流

### 2.1 总体数据流

```
camerad (20Hz, YUV420)
  │ VisionIPC 共享内存
  ▼
modeld
  ├─ 图像预处理 (OpenCL GPU)
  │   ├─ warpPerspective: 透视变换（相机坐标 → 标定坐标 → 模型坐标）
  │   └─ loadYUV: YUV 通道重排为模型输入格式
  │
  ├─ 视觉网络前向 (Tinygrad/QCOM)
  │   ├─ 输入: 主相机帧 img (1,12,128,256) + 广角帧 big_img (1,12,128,256)
  │   └─ 输出: 1576 维向量 → 解析为 pose, 车道线, 前车, hidden_state 等
  │
  ├─ 策略网络前向 (Tinygrad/QCOM)
  │   ├─ 输入: features_buffer (1,25,512) + desire_pulse (1,25,8) + traffic_convention (1,2)
  │   └─ 输出: 1000 维向量 → 解析为 plan (33×15) + desire_state
  │
  └─ 消息发布
      ├─ modelV2         → plannerd, controlsd, selfdrived, UI
      ├─ drivingModelData → controlsd (简化版)
      └─ cameraOdometry  → locationd (传感器融合)
```

### 2.2 与上下游模块的关系

| 上游模块 | 提供的数据 | 用途 |
|----------|-----------|------|
| camerad | 主/广角相机帧 (VisionIPC) | 视觉输入 |
| calibrationd | rpyCalib (liveCalibration) | 计算透视变换矩阵 |
| driverMonitoringState | isRHD (左/右舵) | traffic_convention 输入 |
| carState | vEgo (车速) | 曲率计算、低速保护 |
| carControl | latActive | DesireHelper 状态更新 |
| liveDelay | lateralDelay | 横向动作时间补偿 |
| CarParams | longitudinalActuatorDelay | 纵向动作时间补偿 |

| 下游模块 | 订阅的消息 | 使用的字段 |
|----------|-----------|-----------|
| plannerd | modelV2 | position, velocity, acceleration → 纵向规划 |
| controlsd | modelV2 + drivingModelData | action.desiredCurvature/desiredAcceleration → 横纵向控制 |
| locationd | cameraOdometry | trans, rot → EKF 传感器融合 |
| calibrationd | cameraOdometry | trans (速度) → 标定状态判定 |
| selfdrived | modelV2 | meta, confidence → 系统状态管理 |
| UI | modelV2 | laneLines, roadEdges, leadsV3, position → 可视化 |

## 3. 输入数据

### 3.1 图像输入

modeld 通过 VisionIPC 从 camerad 接收两路相机流：

- **主路相机 (ROAD)**: 窄视场角 (FOV)，高焦距，用于远距离感知
- **广角相机 (WIDE_ROAD)**: 宽视场角，低焦距，用于近距离和侧向感知

在仅有单路相机的设备上，两路输入使用相同的广角流。

**原始帧格式**: YUV420，分辨率因硬件而异（如 TICI 上 AR0231 传感器为 1928×1208）

**模型输入格式**: 每路相机经过透视变换和 YUV 重排后，产生 `(1, 12, 128, 256)` 的 uint8 张量：
- 12 通道 = 2 帧 × 6 通道/帧
- 6 通道 = 4 个 Y 子采样 + U + V
- 2 帧是时间间隔的帧对（当前帧和 `temporal_skip` 帧之前的帧）

### 3.2 图像预处理流水线

图像从原始 YUV420 到模型输入，经历以下 GPU 处理步骤：

#### 3.2.1 透视变换 (warpPerspective)

**目的**: 将相机图像从物理相机坐标系变换到标定后的模型输入坐标系，消除相机安装偏差的影响。

**变换矩阵计算** (`common/transformations/model.py:get_warp_matrix()`):

```
warp_matrix = camera_intrinsics @ view_from_device @ device_from_calib @ calib_from_model
```

其中：
- `camera_intrinsics`: 物理相机内参矩阵 (焦距、光心)
- `view_from_device`: 设备坐标系到视图坐标系的旋转
- `device_from_calib`: 标定欧拉角 (rpyCalib) 对应的旋转矩阵
- `calib_from_model`: 模型虚拟内参的逆矩阵

主相机使用 MED 模型参数（焦距 910.0，分辨率 512×256），广角使用 SBIG 模型参数（焦距 455.0，分辨率 512×256）。

**OpenCL 实现** (`selfdrive/modeld/transforms/transform.cl`):

内核对每个输出像素执行：
1. 应用 3×3 透视变换矩阵，计算源像素坐标
2. 双线性插值采样 4 个邻近像素
3. 分别对 Y、U、V 三个通道执行（3 次 kernel dispatch）

Y 通道工作量: 512×256 = 131,072 个工作项
U/V 通道工作量: 各 256×128 = 32,768 个工作项

#### 3.2.2 YUV 重排 (loadYUV)

**目的**: 将透视变换后的 YUV420 平面数据重排为模型期望的交织格式。

**OpenCL 实现** (`selfdrive/modeld/transforms/loadyuv.cl`):

`loadys` 内核将 Y 通道按 2×2 块拆分为 4 个子采样通道：
```
Y[0::2, 0::2] → 通道 0
Y[0::2, 1::2] → 通道 2
Y[1::2, 0::2] → 通道 1
Y[1::2, 1::2] → 通道 3
```

`loaduv` 内核直接拷贝 U、V 通道到通道 4、5。

#### 3.2.3 时序帧管理

`DrivingModelFrame` 类（`selfdrive/modeld/models/commonmodel.cc`）维护一个环形缓冲区 `img_buffer_20hz_cl`，容量为 `(temporal_skip + 1)` 帧。每次 `prepare()` 调用时：

1. 对新帧执行透视变换 + YUV 重排
2. 将缓冲区中的旧帧向前移动一个位置
3. 新帧写入缓冲区末尾
4. 从缓冲区首帧和末帧组合为模型的 2 帧输入

`temporal_skip = MODEL_RUN_FREQ / MODEL_CONTEXT_FREQ - 1 = 20/5 - 1 = 3`，即每 4 帧取一帧作为历史帧，使得两帧间隔 0.2 秒（对应 5Hz 的模型上下文频率）。

### 3.3 非图像输入

| 输入名 | 形状 | 描述 |
|--------|------|------|
| desire_pulse | (1, 25, 8) | 驾驶意图脉冲序列，8 维 one-hot 编码 |
| traffic_convention | (1, 2) | 交通规则，[1,0]=左舵，[0,1]=右舵 |
| features_buffer | (1, 25, 512) | 视觉网络输出的隐藏状态时间序列 |

#### 3.3.1 Desire 脉冲机制

modeld 内部对 desire 输入采用**上升沿脉冲**机制，而非持续信号：

```python
new_desire = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
```

只有当 desire 从 0 变为 1 的瞬间才产生脉冲（值为 1），其余时刻为 0。这意味着模型自身负责追踪动作的完成，而非依赖外部的持续输入信号。

Desire 的 8 种类型：
```
0: none (无意图)
1: turnLeft (左转)
2: turnRight (右转)
3: laneChangeLeft (左变道)
4: laneChangeRight (右变道)
5: keepLeft (靠左)
6: keepRight (靠右)
7: (保留)
```

#### 3.3.2 InputQueues 滑动窗口

由于视觉网络以 20Hz 运行但策略网络使用 5Hz 的上下文频率，`InputQueues` 类管理一个滑动窗口来对齐时间尺度：

- **图像通道**: 在窗口内按等间隔抽取 `N_FRAMES=2` 帧
- **脉冲信号**: 在对应时间区间内取最大值（只要区间内有脉冲即为 1）
- **特征向量**: 按 `env_fps/model_fps = 4` 的间隔采样

窗口长度为 25（= 5 秒 × 5Hz），存储策略网络所需的完整时间上下文。

### 3.4 TICI 显存零拷贝优化

在 TICI 硬件（Qualcomm 芯片）上，OpenCL 预处理的输出缓冲区通过 `qcom_tensor_from_opencl_address()` 直接映射为 Tinygrad 张量，避免了 GPU → CPU → GPU 的数据拷贝：

```python
# selfdrive/modeld/runners/tinygrad_helpers.py
def qcom_tensor_from_opencl_address(opencl_address, shape, dtype):
    cl_buf_desc_ptr = to_mv(opencl_address, 8).cast('Q')[0]
    rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20]  # GPU 原始指针
    return Tensor.from_blob(rawbuf_ptr, shape, dtype=dtype, device='QCOM')
```

在非 TICI 平台（x86 开发环境）上，则通过 `buffer_from_cl()` 读回 CPU 再构造张量。

## 4. 神经网络架构

### 4.1 双网络设计

modeld 使用两个独立的神经网络协同工作：

#### 视觉网络 (Vision Network)
- **模型文件**: `selfdrive/modeld/models/driving_vision_tinygrad.pkl`
- **元数据**: `selfdrive/modeld/models/driving_vision_metadata.pkl`
- **输入**: 主相机帧 `img` + 广角帧 `big_img`（各 `(1, 12, 128, 256)` uint8）
- **输出**: 1576 维 float32 向量
- **功能**: 图像特征提取、感知（车道线、前车、路沿）、相机运动估计

#### 策略网络 (Policy Network)
- **模型文件**: `selfdrive/modeld/models/driving_policy_tinygrad.pkl`
- **元数据**: `selfdrive/modeld/models/driving_policy_metadata.pkl`
- **输入**: `features_buffer (1,25,512)` + `desire_pulse (1,25,8)` + `traffic_convention (1,2)`
- **输出**: 1000 维 float32 向量
- **功能**: 基于历史视觉特征的行驶轨迹规划

### 4.2 推理执行流程

每次 `ModelState.run()` 调用：

```
1. desire 上升沿检测 → 生成脉冲
2. DrivingModelFrame.prepare() × 2 路 → OpenCL 预处理
3. [丢帧检测] 若 prepare_only=True → 跳过前向，返回 None
4. vision_run(**vision_inputs) → 1576 维输出
5. slice_outputs() → 按元数据切分为命名子向量
6. parse_vision_outputs() → 统计量解码（softmax, sigmoid, MDN）
7. hidden_state (512维) → 入队 features_buffer
8. policy_run(**policy_inputs) → 1000 维输出
9. parse_policy_outputs() → 规划解码
10. 合并视觉 + 策略输出 → 返回
```

### 4.3 推理后端

运行时推理引擎由环境变量 `DEV` 控制：
- **TICI** (comma 3X): `DEV=QCOM`，使用 Qualcomm Adreno GPU (OpenCL)
- **x86 开发环境**: `DEV=CPU`，CPU 推理
- **USB GPU**: `DEV=AMD`，外接 AMD GPU

模型以 pickle 格式序列化（Tinygrad 计算图），加载后直接调用执行前向传播。

## 5. 输出数据

### 5.1 输出解析

模型的原始输出是扁平浮点向量，需要通过 `Parser` 类（`selfdrive/modeld/parse_model_outputs.py`）解码为结构化数据。

主要解码方法：
- **MDN (Mixture Density Network)**: 解码为均值 μ 和标准差 σ，用于位置、速度等连续量
- **Softmax**: 解码为分类概率分布，用于意图预测
- **Sigmoid**: 解码为独立概率，用于事件预测

### 5.2 视觉网络输出 (1576 维)

| 输出名 | 切片范围 | 维度 | 解码方式 | 描述 |
|--------|---------|------|---------|------|
| meta | 0:55 | 55 | sigmoid | 元事件预测（脱离概率、急刹概率等） |
| desire_pred | 55:87 | 32=4×8 | softmax | 未来 4 个时段的意图分布 |
| pose | 87:99 | 12=6×2 | MDN | 相机位姿（平移 xyz + 旋转 rpy，含 std） |
| wide_from_device_euler | 99:105 | 6=3×2 | MDN | 广角-设备欧拉角 |
| road_transform | 105:117 | 12=6×2 | MDN | 路面坐标变换 |
| lane_lines | 117:645 | 528 | MDN | 4 条车道线的 (y,z) 序列 |
| lane_lines_prob | 645:653 | 8 | sigmoid | 车道线存在概率 |
| road_edges | 653:917 | 264 | MDN | 2 条路沿线的 (y,z) 序列 |
| lead | 917:1061 | 144 | MDN+MHP | 前车状态预测 |
| lead_prob | 1061:1064 | 3 | sigmoid | 3 个前车目标的存在概率 |
| hidden_state | 1064:1576 | 512 | 直接传递 | 视觉特征，传递给策略网络 |

### 5.3 策略网络输出 (1000 维)

| 输出名 | 切片范围 | 维度 | 解码方式 | 描述 |
|--------|---------|------|---------|------|
| plan | 0:990 | 990 | MDN+MHP | 行驶规划：33 个时间点 × 15 维状态 |
| desire_state | 990:998 | 8 | softmax | 当前意图状态分布 |

### 5.4 Plan 的 15 维状态量

plan 输出的每个时间点包含 15 个物理量，按 `Plan` 类定义：

| 切片 | 维度 | 含义 |
|------|------|------|
| POSITION (0:3) | x, y, z | 未来位置（米），x=前方，y=左方，z=上方 |
| VELOCITY (3:6) | vx, vy, vz | 速度（m/s） |
| ACCELERATION (6:9) | ax, ay, az | 加速度（m/s²） |
| T_FROM_CURRENT_EULER (9:12) | roll, pitch, yaw | 相对当前姿态的欧拉角变化 |
| ORIENTATION_RATE (12:15) | roll_rate, pitch_rate, yaw_rate | 角速度（rad/s） |

时间索引 `T_IDXS` 为非均匀分布的 33 个时间点，覆盖 0~10 秒，近处密集、远处稀疏：
```python
T_IDXS[i] = 10.0 * (i/32)²
# T_IDXS ≈ [0, 0.01, 0.04, 0.09, 0.16, ..., 7.81, 8.79, 9.77, 10.0]
```

空间索引 `X_IDXS` 类似分布，覆盖 0~192 米。

### 5.5 MHP (Multi-Hypothesis Prediction)

plan 和 lead 使用多假设预测：

- **Plan**: 5 个假设（`PLAN_MHP_N=5`），选择权重最高的 1 个（`PLAN_MHP_SELECTION=1`）
- **Lead**: 2 个假设（`LEAD_MHP_N=2`），选择权重最高的 3 个目标（`LEAD_MHP_SELECTION=3`）

每个假设包含均值 μ 和标准差 σ，以及一个 softmax 权重用于排序。

### 5.6 Meta 事件预测

meta 输出的 55 维向量包含丰富的事件预测：

| 切片 | 含义 |
|------|------|
| ENGAGED (0:1) | 当前接管概率 |
| GAS_DISENGAGE (1:31:6) | 油门导致脱离概率（2/4/6/8/10 秒） |
| BRAKE_DISENGAGE (2:31:6) | 制动导致脱离概率 |
| STEER_OVERRIDE (3:31:6) | 方向盘接管概率 |
| HARD_BRAKE_3 (4:31:6) | 3m/s² 急刹概率 |
| HARD_BRAKE_4 (5:31:6) | 4m/s² 急刹概率 |
| HARD_BRAKE_5 (6:31:6) | 5m/s² 急刹概率 |
| GAS_PRESS (31:55:4) | 油门踏板按压概率 |
| BRAKE_PRESS (32:55:4) | 制动踏板按压概率 |
| LEFT_BLINKER (33:55:4) | 左转灯概率 |
| RIGHT_BLINKER (34:55:4) | 右转灯概率 |

## 6. 消息发布

modeld 每次成功推理后发布 3 条 cereal 消息：

### 6.1 modelV2

完整的模型输出消息，所有下游模块的主要数据来源。包含：

**轨迹规划** (33 个时间点的 XYZ 序列):
- `position`: 未来位置
- `velocity`: 未来速度
- `acceleration`: 未来加速度
- `orientation`: 未来姿态角
- `orientationRate`: 未来角速度

**环境感知**:
- `laneLines[4]`: 4 条车道线几何（沿 X_IDXS 的 y,z 值）
- `laneLineProbs[4]`: 车道线置信度
- `roadEdges[2]`: 2 条路沿线几何
- `leadsV3[3]`: 3 个前车目标的 (x, y, v, a) 时间序列

**元信息** (`meta`):
- `desireState[8]`: 当前意图分布
- `desirePrediction[32]`: 未来意图预测
- `engagedProb`: 接管概率
- `disengagePredictions`: 多种脱离事件概率
- `hardBrakePredicted`: 前碰撞预警（FCW）
- `laneChangeState/Direction`: 变道状态机

**控制动作** (`action`):
- `desiredCurvature`: 期望曲率 (1/m)
- `desiredAcceleration`: 期望加速度 (m/s²)
- `shouldStop`: 是否应停车

**置信度** (`confidence`):
- `green`: 正常
- `yellow`: 中等置信度
- `red`: 低置信度

### 6.2 drivingModelData

简化版消息，包含：
- `path`: 4 次多项式路径系数（对 plan 中 position 的拟合）
- `laneLineMeta`: 左右车道线 y 值和概率
- `action`: 同 modelV2
- `meta`: 变道状态

### 6.3 cameraOdometry

相机运动估计消息，供 locationd 传感器融合使用：
- `trans[3]`: 平移速度（设备坐标系 m/s）
- `rot[3]`: 旋转角速度（rad/s）
- `transStd[3]`, `rotStd[3]`: 标准差
- `wideFromDeviceEuler[3]`: 广角相机相对设备的欧拉角（供 calibrationd 使用）
- `roadTransformTrans[3]`: 路面坐标变换

## 7. 动作生成

### 7.1 从规划到控制指令

`get_action_from_model()` 函数从模型规划中提取控制动作：

#### 期望加速度

```python
# get_accel_from_plan(): 从速度序列反推加速度
v_target = np.interp(action_t, T_IDXS, plan_velocities)  # 插值到 action_t 时刻
a_target = 2 * (v_target - v_now) / action_t - a_now      # 运动学反推
```

其中 `action_t = long_delay + DT_MDL`，考虑了：
- `CP.longitudinalActuatorDelay`: 车辆纵向执行器延迟（品牌相关）
- `LONG_SMOOTH_SECONDS = 0.3`: 纵向平滑时间常数
- `DT_MDL = 0.05`: 模型周期

#### 期望曲率

```python
# get_curvature_from_plan(): 从航向角序列反推曲率
psi_target = np.interp(action_t, T_IDXS, plan_yaws)
curvature = 2 * psi_target / (v_ego * action_t) - yaw_rate / v_ego
```

其中 `action_t = lat_delay + DT_MDL`，`lat_delay` 来自 `liveDelay` 消息。

### 7.2 平滑滤波

动作输出经一阶低通滤波平滑：

```python
# smooth_value(): 一阶指数平滑
alpha = 1 - exp(-dt / tau)
smoothed = alpha * new_value + (1 - alpha) * prev_value
```

- 横向: `LAT_SMOOTH_SECONDS = 0.0`（无平滑，即直接输出）
- 纵向: `LONG_SMOOTH_SECONDS = 0.3`

### 7.3 低速保护

当车速低于 `MIN_LAT_CONTROL_SPEED = 0.3 m/s` 时，保持上一时刻的曲率值，避免低速抖动。

## 8. 前碰撞预警 (FCW)

modeld 内置了基于概率的前碰撞预警机制：

```python
# fill_model_msg.py
hard_brake_predicted = (
    prev_brake_5ms2_probs > FCW_THRESHOLDS_5MS2  # [.05, .05, .15, .15, .15]
).all() and (
    prev_brake_3ms2_probs > FCW_THRESHOLDS_3MS2  # [.7, .7]
).all()
```

当最近 5 帧的 5m/s² 急刹概率和最近 2 帧的 3m/s² 急刹概率同时超过阈值时，触发 `hardBrakePredicted`。

## 9. 置信度评估

modeld 基于脱离事件预测计算系统置信度：

1. 每 2 秒从 meta 输出中提取综合脱离概率（油门/制动/方向盘接管）
2. 将独立脱离概率推入长度为 5 的滚动缓冲区
3. 计算加权得分，映射到三级置信度：
   - `green`: score < 0.01165（正常）
   - `yellow`: score < 0.06157（注意）
   - `red`: score ≥ 0.06157（警告）

## 10. DesireHelper 变道状态机

`DesireHelper`（`selfdrive/controls/lib/desire_helper.py`）管理变道意图，将转向灯信号转换为模型输入。

### 状态转换

```
off → preLaneChange（转向灯打开 + 速度 > 20 MPH）
  → laneChangeStarting（转向灯方向受力 + 无盲区障碍）
  → laneChangeFinishing（lane_change_prob < 0.02，模型认为变道完成）
  → off
```

### 超时保护

- `LANE_CHANGE_TIME_MAX = 10s`: 变道超时自动取消

### 车道线概率渐变

- `laneChangeStarting`: 0.5 秒内将 `lane_change_ll_prob` 从 1.0 渐出到 0
- `laneChangeFinishing`: 1 秒内渐入回 1.0

## 11. 丢帧处理

modeld 通过 VisionIPC 帧序号检测丢帧：

```python
vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
```

当检测到丢帧时：
1. `prepare_only = True`：仅执行图像预处理（更新 OpenCL 缓冲区），跳过网络前向
2. 丢帧计数通过一阶滤波器追踪：`frame_dropped_filter = FirstOrderFilter(0., 10., 1/20Hz)`
3. `frame_drop_ratio` 写入输出消息，供下游判断数据质量

前 10 帧不计入丢帧统计（预热期）。

## 12. 坐标系定义

modeld 涉及多个坐标系的转换：

| 坐标系 | X 轴 | Y 轴 | Z 轴 | 用途 |
|--------|------|------|------|------|
| 设备坐标系 (Device) | 前方 | 右方 | 下方 | 物理传感器安装参考 |
| 视图坐标系 (View) | 右方 | 下方 | 前方 | 图像处理中间坐标 |
| 标定坐标系 (Calib) | 前方 | 左方 | 上方 | 消除安装偏差后的参考系 |
| 模型坐标系 (Model) | — | — | — | 模型虚拟相机的像素坐标 |

模型输出的 plan 中 position/velocity/acceleration 均在标定坐标系下表述：
- x: 前方距离/速度/加速度
- y: 左方偏移
- z: 高度

## 13. 关键配置参数

| 参数 | 值 | 定义位置 | 含义 |
|------|-----|---------|------|
| MODEL_RUN_FREQ | 20 Hz | constants.py | 模型运行频率 |
| MODEL_CONTEXT_FREQ | 5 Hz | constants.py | 模型上下文频率（策略网络时序分辨率） |
| N_FRAMES | 2 | constants.py | 视觉输入的时序帧数 |
| FEATURE_LEN | 512 | constants.py | 隐藏状态维度 |
| DESIRE_LEN | 8 | constants.py | 意图类型数量 |
| IDX_N | 33 | constants.py | 时间/空间索引点数 |
| PLAN_MHP_N | 5 | constants.py | plan 多假设数量 |
| LEAD_MHP_N | 2 | constants.py | lead 多假设数量 |
| DT_MDL | 0.05 s | realtime.py | 模型帧间隔 (1/20Hz) |

## 14. 关键文件索引

| 文件 | 功能 |
|------|------|
| `selfdrive/modeld/modeld.py` | 主入口，主循环，模型加载与推理 |
| `selfdrive/modeld/fill_model_msg.py` | cereal 消息填充 |
| `selfdrive/modeld/parse_model_outputs.py` | 网络输出解析（MDN, softmax, sigmoid） |
| `selfdrive/modeld/constants.py` | 常量定义（时间索引、输出维度、阈值） |
| `selfdrive/modeld/models/commonmodel.h/cc` | C++ 帧处理类（DrivingModelFrame） |
| `selfdrive/modeld/transforms/transform.cl` | OpenCL 透视变换内核 |
| `selfdrive/modeld/transforms/transform.cc` | 透视变换 C++ 调度（Y/U/V 三次 dispatch） |
| `selfdrive/modeld/transforms/loadyuv.cl` | OpenCL YUV 重排内核 |
| `selfdrive/modeld/transforms/loadyuv.cc` | YUV 重排 C++ 调度 |
| `selfdrive/modeld/runners/tinygrad_helpers.py` | QCOM 显存零拷贝映射 |
| `common/transformations/model.py` | 变换矩阵计算（get_warp_matrix） |
| `common/transformations/camera.py` | 相机内参、坐标系定义 |
| `selfdrive/controls/lib/desire_helper.py` | 变道状态机 |
| `selfdrive/controls/lib/drive_helpers.py` | 规划→动作转换辅助函数 |
