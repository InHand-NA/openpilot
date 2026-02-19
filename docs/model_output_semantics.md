# supercombo 模型输出数据语义参考

本文档详细描述 openpilot supercombo 双网络模型的架构、每个输出字段的物理语义、数值单位和坐标系约定。

**相关源码**:
- 模型推理主循环: `selfdrive/modeld/modeld.py`
- 输出解析: `selfdrive/modeld/parse_model_outputs.py`
- 输出填充: `selfdrive/modeld/fill_model_msg.py`
- 常量定义: `selfdrive/modeld/constants.py`
- 消息定义: `cereal/log.capnp` (ModelDataV2, DrivingModelData, CameraOdometry)
- 模型坐标变换: `common/transformations/model.py`
- 动作推导: `selfdrive/controls/lib/drive_helpers.py`

---

## 1. 双网络架构总览

supercombo 模型由两个独立的神经网络组成，通过隐藏状态向量串联：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     supercombo 双网络架构                                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ 视觉网络 (driving_vision)                                       │     │
│  │                                                                 │     │
│  │  输入:                                                          │     │
│  │    img      (1, 12, 128, 256)  ← 窄FOV主摄 (fcam/MEDModel)    │     │
│  │    big_img  (1, 12, 128, 256)  ← 广FOV副摄 (ecam/SBIGModel)   │     │
│  │                                                                 │     │
│  │  输出: 1576 维向量                                               │     │
│  │    ├─ pose (6D)           → cameraOdometry 消息                 │     │
│  │    ├─ wide_from_device_euler (3D) → cameraOdometry              │     │
│  │    ├─ road_transform (6D) → cameraOdometry                     │     │
│  │    ├─ lane_lines (4×33×2) → modelV2.laneLines                  │     │
│  │    ├─ lane_lines_prob (8) → modelV2.laneLineProbs              │     │
│  │    ├─ road_edges (2×33×2) → modelV2.roadEdges                  │     │
│  │    ├─ lead (2×6×4)        → modelV2.leadsV3                    │     │
│  │    ├─ lead_prob (3)       → modelV2.leadsV3[i].prob            │     │
│  │    ├─ meta (55)           → modelV2.meta                       │     │
│  │    ├─ desire_pred (4×8)   → modelV2.meta.desirePrediction      │     │
│  │    └─ hidden_state (512)  ─────────────┐                       │     │
│  └─────────────────────────────────────────│───────────────────────┘     │
│                                            │                             │
│                                            ▼                             │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ 策略网络 (driving_policy)                                       │     │
│  │                                                                 │     │
│  │  输入:                                                          │     │
│  │    features_buffer      (1, 25, 512)  ← 滑动窗口历史隐藏状态   │     │
│  │    desire_pulse         (1, 25, 8)    ← 驾驶意图脉冲序列       │     │
│  │    traffic_convention   (1, 2)        ← 左/右手驾驶独热编码     │     │
│  │                                                                 │     │
│  │  输出: 1000 维向量                                               │     │
│  │    ├─ plan (5×33×15)      → modelV2.position/velocity/...      │     │
│  │    └─ desire_state (8)    → modelV2.meta.desireState            │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  后处理:                                                                  │
│    plan → get_action_from_model() → Action{curvature, accel, stop}       │
│                                                                          │
│  发布消息:                                                                │
│    pm.send('modelV2')          ← 视觉+策略合并输出                       │
│    pm.send('drivingModelData') ← 简化控制数据 (Action + path + meta)     │
│    pm.send('cameraOdometry')   ← 视觉网络自运动估计                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.1 关键设计分离

| 属性 | 视觉网络 | 策略网络 |
|------|----------|----------|
| **职责** | 环境感知（几何+语义） | 行为规划（轨迹+决策） |
| **输入** | 双目图像帧 (uint8) | 视觉特征 + 意图 + 交规 (float32) |
| **输出维度** | 1576 维 | 1000 维 |
| **耦合点** | hidden_state (512D) 写出 | features_buffer (512D) 读入 |
| **时间上下文** | 2 帧图像（当前帧 + 200ms 前帧） | 25 帧历史特征（~5 秒） |
| **推理频率** | 20 Hz（与策略网络同步） | 20 Hz |
| **有效输出频率** | 20 Hz | 20 Hz |

### 1.2 数据流时序

```python
# modeld.py 主循环中的每一帧:

# 1. 视觉前向
vision_output = vision_run(img=..., big_img=...)           # → 1576D
vision_dict = parser.parse_vision_outputs(vision_output)   # 解析为命名字典

# 2. 隐藏状态入队（连接双网络的桥梁）
input_queues.enqueue({
  'features_buffer': vision_dict['hidden_state'],   # (1, 1, 512)
  'desire_pulse': new_desire                         # (1, 1, 8)
})

# 3. 策略前向
policy_output = policy_run(
  features_buffer=...,    # (1, 25, 512) 滑动窗口
  desire_pulse=...,       # (1, 25, 8)   滑动窗口
  traffic_convention=...  # (1, 2)       静态
)                                                          # → 1000D
policy_dict = parser.parse_policy_outputs(policy_output)

# 4. 合并输出
combined = {**vision_dict, **policy_dict}

# 5. 动作推导 + 消息发布
action = get_action_from_model(combined, ...)
fill_model_msg(...)      # → modelV2 + drivingModelData
fill_pose_msg(...)       # → cameraOdometry
```

---

## 2. 坐标系

模型的所有 3D 输出（位置、速度、加速度、车道线、路沿、前车）都在**标定坐标系 (Calibrated Frame)** 中表述。

### 2.1 标定坐标系定义

标定坐标系是消除了相机安装偏差后的参考系。它与车辆坐标系 (Car Frame) 在 pitch 和 yaw 上对齐，与设备坐标系 (Device Frame) 在 roll 上对齐，原点位于设备（相机）处。

**轴定义**（参见 `common/transformations/README.md`）:

| 轴 | 方向 | 正值含义 |
|----|------|---------|
| **x** | 前方 (Forward) | 离车越远，值越大 |
| **y** | 右方 (Right) | 车辆右侧为正 |
| **z** | 下方 (Down) | 路面方向为正 |

> **注意**: capnp schema 注释写着 "All SI units and in device frame"，但 Device Frame 和 Calibrated Frame 都是 [Forward, Right, Down]，仅通过 rpyCalib 旋转对齐。模型的 warp 矩阵已经将图像变换到标定坐标系，因此模型输出本质上在标定坐标系中。

### 2.2 相关坐标系对比

| 坐标系 | x | y | z | 用途 |
|--------|---|---|---|------|
| Device | 前 | 右 | 下 | 物理传感器安装 |
| Calibrated | 前 | 右 | 下 | 模型输出、控制计算 |
| View | 右 | 下 | 前 | 图像投影中间坐标 |
| Car | 前 | 右 | 下 | 与路面和车辆方向对齐 |

### 2.3 投影关系

从标定坐标系 3D 点投影到图像像素的变换链:

```
pixel = K @ view_frame_from_device_frame @ device_from_calib @ point_3d
```

其中 `view_frame_from_device_frame` 将 [x_fwd, y_right, z_down] 重排为 [y_right, z_down, x_fwd]，再由内参矩阵 K 投影到像素。透视除法后:

```
u = fx * y / x + cx    (水平像素位置)
v = fx * z / x + cy    (垂直像素位置)
```

因此 **z/x 比值直接决定了 3D 点在图像中的垂直位置**（v 坐标）。

### 2.4 Warp 矩阵（图像 → 模型坐标系）

模型不直接处理原始相机图像，而是通过 `get_warp_matrix()` 将图像变换到标定坐标系下的虚拟相机视角:

```python
# common/transformations/model.py
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame):
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model  # 3×3 单应矩阵
    return warp_matrix
```

**模型虚拟相机参数**:

| 模型 | 分辨率 | 焦距 (px) | CY 偏移 | 用途 |
|------|--------|-----------|---------|------|
| MEDModel | 512×256 | 910.0 | 47.6 | 主路 fcam（窄 FOV） |
| SBIGModel | 512×256 | 455.0 | 151.8 | 辅路 ecam（广 FOV） |

> MEDModel 焦距 910px 在 512px 宽度上对应 ~31° 半视场角；SBIGModel 焦距 455px 对应 ~60° 半视场角。CY 偏移使虚拟相机"抬头"，增加地面覆盖范围。

---

## 3. 采样网格

模型输出不是对均匀网格的预测，而是在二次函数分布的采样点上。

### 3.1 时间索引 T_IDXS

用于 plan 相关输出（position, velocity, acceleration, orientation, orientationRate）。

```python
T_IDXS[i] = 10.0 × (i/32)²    # i = 0, 1, ..., 32
```

33 个时间点，覆盖 **0 到 10 秒**，近处密集、远处稀疏:

| 索引 | 0 | 1 | 2 | 5 | 8 | 16 | 24 | 32 |
|------|---|---|---|---|---|----|----|-----|
| 时间(s) | 0.0 | 0.010 | 0.039 | 0.244 | 0.625 | 2.5 | 5.625 | 10.0 |

**物理意义**: 预测自我车辆未来 10 秒的状态。近处采样密（~10ms 间隔），远处采样疏（~1.2s 间隔），反映了对近期精确性和远期趋势的不同需求。

### 3.2 距离索引 X_IDXS

用于车道线 (laneLines) 和路沿 (roadEdges) 的空间采样。

```python
X_IDXS[i] = 192.0 × (i/32)²    # i = 0, 1, ..., 32
```

33 个距离点，覆盖 **0 到 192 米**:

| 索引 | 0 | 1 | 2 | 5 | 8 | 16 | 24 | 32 |
|------|---|---|---|---|---|----|----|-----|
| 距离(m) | 0.0 | 0.188 | 0.75 | 4.69 | 12.0 | 48.0 | 108.0 | 192.0 |

**物理意义**: 在不同前方距离处采样车道线和路沿的横向偏移和高度。近处密采保证近处几何精度，远处稀疏减少不确定性累积。

### 3.3 前车时间索引 LEAD_T_IDXS

用于前车预测 (leadsV3)。

```python
LEAD_T_IDXS = [0., 2., 4., 6., 8., 10.]    # 6 个时间点
```

均匀分布，每 2 秒一个采样点，共 **6 个点**覆盖 0-10 秒。

### 3.4 元事件时间索引 META_T_IDXS

用于脱离预测 (disengagePredictions)。

```python
META_T_IDXS = [2., 4., 6., 8., 10.]    # 5 个时间点
```

---

## 4. 视觉网络输出

视觉网络 (driving_vision) 处理双目图像，输出 **1576 维**向量。解析由 `Parser.parse_vision_outputs()` 完成。

### 4.1 输出切片布局

视觉网络的 1576 维扁平输出通过元数据定义的切片拆分为命名子向量:

| 输出名称 | 维度 | 解码方式 | 归属消息 |
|---------|------|---------|---------|
| `meta` | 55 | sigmoid (BCE) | modelV2.meta |
| `desire_pred` | 32 (4×8) | softmax (CCE) | modelV2.meta.desirePrediction |
| `pose` | 12 (6×2) | MDN | cameraOdometry.trans/rot |
| `wide_from_device_euler` | 6 (3×2) | MDN | cameraOdometry.wideFromDeviceEuler |
| `road_transform` | 12 (6×2) | MDN | cameraOdometry.roadTransformTrans |
| `lane_lines` | 528 (4×33×2×2) | MDN | modelV2.laneLines |
| `lane_lines_prob` | 8 | sigmoid (BCE) | modelV2.laneLineProbs |
| `road_edges` | 264 (2×33×2×2) | MDN | modelV2.roadEdges |
| `lead` | 144+ | MDN + MHP(2) | modelV2.leadsV3 |
| `lead_prob` | 3 | sigmoid (BCE) | modelV2.leadsV3[i].prob |
| `hidden_state` | **512** | 直通（不解码） | → 策略网络 features_buffer |

> **hidden_state** 占据输出向量的最后 512 维 (slice 1064:1576)，是视觉网络到策略网络的唯一信息通道。它不发布到任何 cereal 消息中。

### 4.2 Camera Odometry — 自运动估计 (pose)

**维度**: 6 (3 平移 + 3 旋转)，MDN 解码后得到均值和标准差各 6 维。

**发布消息**: `cameraOdometry`（独立消息，不在 modelV2 内）

| 字段 | 源切片 | 类型 | 单位 | 描述 |
|------|--------|------|------|------|
| `trans` | pose[0:3] | List(Float32) × 3 | **m/s** | 平移速度 [tx, ty, tz] |
| `rot` | pose[3:6] | List(Float32) × 3 | **rad/s** | 旋转角速度 [roll, pitch, yaw] |
| `transStd` | pose_stds[0:3] | List(Float32) × 3 | m/s | 平移标准差 |
| `rotStd` | pose_stds[3:6] | List(Float32) × 3 | rad/s | 旋转标准差 |

**语义细节**:
- **trans 是速度而非位移**。`trans[0]` ≈ 前向车速（可与 carState.vEgo 对比），`trans[2]` 反映相机高度和路面坡度的联合效应。
- **rot 是角速度而非累积角度**。`rot[2]` ≈ 偏航率（yaw rate），正值对应向右转弯。
- 坐标系：**标定坐标系**（calibrated frame），不是设备坐标系。locationd 在融合时通过 `device_from_calib` 旋转矩阵将其转换到设备坐标系。

**填充逻辑** (`fill_pose_msg()`, `fill_model_msg.py:186-191`):
```python
cameraOdometry.trans = net_output_data['pose'][0,:3].tolist()
cameraOdometry.rot = net_output_data['pose'][0,3:].tolist()
cameraOdometry.transStd = net_output_data['pose_stds'][0,:3].tolist()
cameraOdometry.rotStd = net_output_data['pose_stds'][0,3:].tolist()
```

**下游消费者**:
- **locationd**: 作为 EKF 观测输入，与 IMU 陀螺仪/加速度计融合。注意 locationd 会将标准差放大 2~10 倍（`rot_calib_std *= 10; trans_calib_std *= 2`），以补偿时间相关噪声违反 EKF 白噪声假设的问题。
- **calibrationd**: 从 `trans` 估计相机标定参数（需 vEgo > 阈值）。
- **帧间变换**: 短时间窗口（<10s）内逐帧累积 cameraOdometry 的精度优于 livePose（详见 `docs/lane_line_labeling_from_real_vehicles.md`）。

### 4.3 广角-设备欧拉角 (wide_from_device_euler)

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `wideFromDeviceEuler` | List(Float32) × 3 | **rad** | 广角相机相对设备的 [roll, pitch, yaw] |
| `wideFromDeviceEulerStd` | List(Float32) × 3 | rad | 标准差 |

**用途**: 描述广角相机 (ecam) 相对于设备坐标系的旋转偏差。用于双相机几何对齐。

### 4.4 路面变换 (road_transform)

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `roadTransformTrans` | List(Float32) × 3 | **m/s** | 路面坐标变换的平移分量 [x, y, z] |
| `roadTransformTransStd` | List(Float32) × 3 | m/s | 标准差 |

**用途**: calibrationd 从 `roadTransformTrans[2]`（z 分量）估计相机到路面的高度。

### 4.5 车道线 (lane_lines)

模型预测 **4 条车道线**，编号 0-3:
- `laneLines[0]`: 最左侧车道线（远左，即左相邻车道的左边界）
- `laneLines[1]`: 左相邻车道线（ego 车道左边界）
- `laneLines[2]`: 右相邻车道线（ego 车道右边界）
- `laneLines[3]`: 最右侧车道线（远右，即右相邻车道的右边界）

**解码方式**: MDN（均值 + 标准差），无 MHP。

**原始输出维度**: `(4, 33, 2)` — 4 条线 × 33 个 X_IDXS 采样点 × 2 个预测值 (y, z)。MDN 标准差再增加一倍，总共 `4 × 33 × 2 × 2 = 528` 维。

每条车道线是一个 XYZTData:

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `laneLines[i].t` | List(Float32) | — | **空列表**（车道线是空间采样，不是时间采样） |
| `laneLines[i].x` | List(Float32) | **米** | 前方距离（固定为 X_IDXS，33 个点，0~192m） |
| `laneLines[i].y` | List(Float32) | **米** | 每个距离处车道线的**横向偏移**（右方为正） |
| `laneLines[i].z` | List(Float32) | **米** | 每个距离处车道线的**垂直偏移**（下方为正，近似路面高度） |

**x 字段的特殊性**: 车道线的 `x` 不是模型预测的，而是**固定值** X_IDXS——它是采样网格，不是输出。模型实际预测的是每个 X_IDXS 距离处的 `(y, z)` 值。

**y 的语义**: 正值表示车道线在车辆右侧。通常 `laneLines[1].y[0]` 是负值（左车道线在左边），`laneLines[2].y[0]` 是正值（右车道线在右边）。

**z 的语义**: 路面在标定坐标系中的高度（z=down）。对于平坦路面，z 值接近**相机高度**（约 1.22m），因为路面在相机下方。z 值的变化反映路面坡度和起伏。

**填充逻辑** (`fill_model_msg.py:101-107`):
```python
for i in range(4):
    lane_line = modelV2.laneLines[i]
    fill_xyzt(lane_line, LINE_T_IDXS,
              np.array(ModelConstants.X_IDXS),       # x = 固定采样距离
              net_output_data['lane_lines'][0,i,:,0], # y = 横向偏移
              net_output_data['lane_lines'][0,i,:,1]) # z = 垂直偏移
modelV2.laneLineStds = net_output_data['lane_lines_stds'][0,:,0,0].tolist()
```

#### 4.5.1 车道线概率 (laneLineProbs)

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `laneLineProbs` | List(Float32) × 4 | **概率 (0~1)** | 4 条车道线的存在概率 |
| `laneLineStds` | List(Float32) × 4 | **米** | 4 条车道线的位置标准差 |

**原始输出到概率的映射**: 网络输出 `lane_lines_prob` 是 8 个 sigmoid 值。这 8 个值的含义推测为 [不存在₀, 存在₀, 不存在₁, 存在₁, ...] 的成对概率。取**奇数索引** `[1, 3, 5, 7]` 得到 4 条线的存在概率:

```python
# fill_model_msg.py:107
modelV2.laneLineProbs = net_output_data['lane_lines_prob'][0,1::2].tolist()
```

**laneLineStds 的取值**: 只取第一条线的第一个采样点的第一个维度（y）的标准差作为该线的整体标准差代表:

```python
# fill_model_msg.py:106
modelV2.laneLineStds = net_output_data['lane_lines_stds'][0,:,0,0].tolist()
```

#### 4.5.2 LaneLineMeta（简化元数据）

发布在 `drivingModelData.laneLineMeta` 中，只包含 ego 车道的即时信息:

```python
# fill_model_msg.py:52-56
builder.leftY = lane_lines[1].y[0]      # 左车道线在 x=0 处的横向偏移
builder.leftProb = lane_line_probs[1]    # 左车道线存在概率
builder.rightY = lane_lines[2].y[0]     # 右车道线在 x=0 处的横向偏移
builder.rightProb = lane_line_probs[2]   # 右车道线存在概率
```

**下游用途**:
- LDW (车道偏离预警): 当 `laneLineProbs[1] > 0.5` 且 `laneLines[1].y[0]` 距离小于阈值时触发
- UI 可视化: 车道线绘制透明度 = `clip(prob, 0, 0.7)`

### 4.6 路沿 (road_edges)

模型预测 **2 条路沿线**:
- `roadEdges[0]`: 左侧路沿
- `roadEdges[1]`: 右侧路沿

**解码方式**: MDN，无 MHP。

**原始输出维度**: `(2, 33, 2)` — 2 条线 × 33 个 X_IDXS 采样点 × 2 个预测值 (y, z)。

结构与车道线完全相同:

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `roadEdges[i].x` | List(Float32) | **米** | 固定为 X_IDXS（33 个距离点） |
| `roadEdges[i].y` | List(Float32) | **米** | 横向偏移（右方为正） |
| `roadEdges[i].z` | List(Float32) | **米** | 垂直偏移（下方为正） |

**标准差**:

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `roadEdgeStds` | List(Float32) × 2 | **米** | 2 条路沿的位置标准差 |

**下游用途**: UI 绘制透明度 = `clip(1.0 - std, 0, 1.0)`，标准差越小越不透明（越确定）。

### 4.7 前车 (lead)

模型预测 **3 个前车目标**，按相关性排序。每个前车包含未来 10 秒的状态时间序列。

**解码方式**: MDN + MHP（2 个假设，选出 3 个目标）

**原始输出**: `LEAD_MHP_N=2` 个假设，每个假设包含 `LEAD_MHP_SELECTION=3` 个目标的 `6 × 4` 轨迹（6 个时间步 × 4 维 [x, y, v, a]）。通过 softmax 权重排序后取最优。

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `leadsV3[i].prob` | Float32 | **概率 (0~1)** | 该前车存在的概率 |
| `leadsV3[i].probTime` | Float32 | **秒** | 概率对应的时间偏移（0/2/4 秒） |
| `leadsV3[i].t` | List(Float32) | 秒 | LEAD_T_IDXS = [0, 2, 4, 6, 8, 10]，6 个时间点 |
| `leadsV3[i].x` | List(Float32) | **米** | 前向距离（6 个时间点） |
| `leadsV3[i].y` | List(Float32) | **米** | 横向偏移（6 个时间点） |
| `leadsV3[i].v` | List(Float32) | **m/s** | **绝对**速度标量（6 个时间点） |
| `leadsV3[i].a` | List(Float32) | **m/s²** | 速度的导数（6 个时间点） |
| `leadsV3[i].{x,y,v,a}Std` | List(Float32) | 对应 | 各量的标准差 |

**字段语义详解**:

- **x（前向距离）**: 前车相对于自我车辆的纵向距离。`leadsV3[0].x[0]` 是当前时刻最近前车的距离。正值表示前方。
- **y（横向偏移）**: 前车相对于自我车辆的横向位置（y=右方为正）。
- **v（绝对速度）**: capnp schema 注释为 "v absolute norm speed"，是前车的**绝对**速度标量，不是相对速度。下游 radard 通过 `v_rel = lead.v[0] - model_v_ego` 计算相对速度。
- **a（加速度）**: v 的时间导数，即前车的纵向加速度。
- **probTime**: 3 个前车目标的概率分别对应不同的时间偏移（`LEAD_T_OFFSETS = [0, 2, 4]` 秒）。`leadsV3[0].probTime = 0` 表示当前时刻的前车，`leadsV3[1].probTime = 2` 表示 2 秒后的前车（可能是不同的车辆）。

**填充逻辑** (`fill_model_msg.py:119-124`):
```python
for i in range(3):
    lead = modelV2.leadsV3[i]
    fill_xyvat(lead, ModelConstants.LEAD_T_IDXS,
               *net_output_data['lead'][0,i].T,       # x, y, v, a
               *net_output_data['lead_stds'][0,i].T)  # 标准差
    lead.prob = net_output_data['lead_prob'][0,i].tolist()
    lead.probTime = ModelConstants.LEAD_T_OFFSETS[i]
```

**下游用途**:
- radard: 视觉-雷达融合，当 `prob > 0.5` 时视为有效前车
- 纵向规划: 前车距离和速度是 ACC 的关键输入
- UI 可视化: 绘制 chevron 标记，显示距离标签

### 4.8 Meta — 元事件预测

**维度**: 55 维 sigmoid 向量，编码多种驾驶事件概率。

**切片布局** (`constants.py` Meta 类):

```python
class Meta:
  ENGAGED         = slice(0, 1)        # 1 维: 系统参与概率
  # 以下每 6 步取 1 个，覆盖 [2, 4, 6, 8, 10] 秒共 5 个时间点
  GAS_DISENGAGE   = slice(1, 31, 6)    # 5 维: 油门脱离
  BRAKE_DISENGAGE = slice(2, 31, 6)    # 5 维: 制动脱离
  STEER_OVERRIDE  = slice(3, 31, 6)    # 5 维: 转向接管
  HARD_BRAKE_3    = slice(4, 31, 6)    # 5 维: ≥3 m/s² 急刹
  HARD_BRAKE_4    = slice(5, 31, 6)    # 5 维: ≥4 m/s² 急刹
  HARD_BRAKE_5    = slice(6, 31, 6)    # 5 维: ≥5 m/s² 急刹
  # 以下每 4 步取 1 个，覆盖 [0, 2, 4, 6, 8, 10] 秒共 6 个时间点
  GAS_PRESS       = slice(31, 55, 4)   # 6 维: 油门按压
  BRAKE_PRESS     = slice(32, 55, 4)   # 6 维: 制动按压
  LEFT_BLINKER    = slice(33, 55, 4)   # 6 维: 左转灯
  RIGHT_BLINKER   = slice(34, 55, 4)   # 6 维: 右转灯
  # 总计: 1 + 5×6 + 6×4 = 55 维
```

**发布到 modelV2.meta**:

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `meta.engagedProb` | Float32 | **概率 (0~1)** | 驾驶员接管/参与概率 |
| `meta.desirePrediction` | List(Float32) × 32 | **概率分布** | 未来意图预测（4×8 维） |
| `meta.hardBrakePredicted` | Bool | — | 前碰撞预警（FCW）标志 |
| `meta.laneChangeState` | Enum | — | 变道状态（off/pre/starting/finishing） |
| `meta.laneChangeDirection` | Enum | — | 变道方向（none/left/right） |

#### 4.8.1 脱离预测 (disengagePredictions)

预测未来 2/4/6/8/10 秒内各类脱离事件的**累积概率**:

| 字段 | 时间点 | 描述 |
|------|--------|------|
| `brakeDisengageProbs` | [2,4,6,8,10]s | 制动导致脱离 |
| `gasDisengageProbs` | [2,4,6,8,10]s | 油门导致脱离 |
| `steerOverrideProbs` | [2,4,6,8,10]s | 方向盘接管 |
| `brake3MetersPerSecondSquaredProbs` | [2,4,6,8,10]s | ≥3 m/s² 急刹 |
| `brake4MetersPerSecondSquaredProbs` | [2,4,6,8,10]s | ≥4 m/s² 急刹 |
| `brake5MetersPerSecondSquaredProbs` | [2,4,6,8,10]s | ≥5 m/s² 急刹 |
| `gasPressProbs` | [0,2,4,6,8,10]s | 油门踏板按压 |
| `brakePressProbs` | [0,2,4,6,8,10]s | 制动踏板按压 |

#### 4.8.2 FCW（前碰撞预警）判定

```python
# fill_model_msg.py:143-149
# 滚动窗口: 连续 5 帧的 brake_5ms2 和 2 帧的 brake_3ms2
hard_brake_predicted = (
    (prev_brake_5ms2_probs > FCW_THRESHOLDS_5MS2).all() and  # [.05,.05,.15,.15,.15]
    (prev_brake_3ms2_probs > FCW_THRESHOLDS_3MS2).all()      # [.7, .7]
)
```

当连续 5 帧的 ≥5m/s² 急刹概率超过递增阈值，**且**连续 2 帧的 ≥3m/s² 急刹概率超过 0.7 时，触发 `hardBrakePredicted = True`。

### 4.9 Desire Prediction — 未来意图预测

**维度**: 32 维 (4 个未来时间步 × 8 个意图类别)

**解码方式**: softmax (CCE)，每个时间步内 8 个类别归一化

**8 种意图类型**:

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | none | 无特殊意图 |
| 1 | turnLeft | 左转 |
| 2 | turnRight | 右转 |
| 3 | laneChangeLeft | 左变道 |
| 4 | laneChangeRight | 右变道 |
| 5 | keepLeft | 靠左行驶 |
| 6 | keepRight | 靠右行驶 |
| 7 | (保留) | — |

**发布**: `modelV2.meta.desirePrediction` — 展平为 32 维列表，按 `[t₀×8类, t₁×8类, t₂×8类, t₃×8类]` 排列，4 个未来时间步由 `DESIRE_PRED_LEN=4` 决定。

### 4.10 隐藏状态 (hidden_state)

**维度**: 512 维（`FEATURE_LEN = 512`）

**位置**: 视觉输出向量的最后 512 维 (slice 1064:1576)

这是视觉网络与策略网络之间的**唯一信息通道**。它承载了视觉网络对当前场景的浓缩表征，包含了无法通过上述结构化输出（车道线、前车等）完全表达的高层语义信息。

**特征缓冲管理**:

```python
# modeld.py:292-294
# 每帧将新的 512D 特征推入滑动窗口
full_input_queues.enqueue({'features_buffer': vision_dict['hidden_state']})
# 窗口大小: (1, 25, 512) — 25 帧历史 = 约 5 秒 (25/5Hz)
```

策略网络的 `features_buffer` 输入维度为 `(1, 25, 512)`。由于模型运行频率 (20Hz) 高于上下文频率 (5Hz)，`InputQueues` 类负责重采样，使 25 帧窗口中等间隔选取特征帧。

---

## 5. 策略网络输出

策略网络 (driving_policy) 接收视觉特征历史和意图信号，输出 **1000 维**向量。解析由 `Parser.parse_policy_outputs()` 完成。

### 5.1 输出切片布局

| 输出名称 | 维度 | 解码方式 | 归属消息 |
|---------|------|---------|---------|
| `plan` | 990 (5×33×15 → MHP → 33×15) | MDN + MHP(5→1) | modelV2.position/velocity/... |
| `desire_state` | 8 | softmax (CCE) | modelV2.meta.desireState |
| (pad) | 2 | — | 内部填充，不使用 |

### 5.2 Plan — 自我轨迹规划

**维度**: MHP 解码前为 `5 × 33 × 15 × 2 + 5 × 1` ≈ 990 维（5 个假设 × 33 时间步 × 15 维/步 × [均值+标准差] + 5 个假设权重）。MHP 解码后取最优 1 个假设，得到 `(1, 33, 15)` 的规划张量。

**每时间步 15 维结构** (`constants.py` Plan 类):

```python
class Plan:
  POSITION             = slice(0, 3)    # [x, y, z]      位置 (米)
  VELOCITY             = slice(3, 6)    # [vx, vy, vz]   速度 (m/s)
  ACCELERATION         = slice(6, 9)    # [ax, ay, az]   加速度 (m/s²)
  T_FROM_CURRENT_EULER = slice(9, 12)   # [roll, pitch, yaw]  相对当前姿态的欧拉角 (rad)
  ORIENTATION_RATE     = slice(12, 15)  # [ωroll, ωpitch, ωyaw]  角速度 (rad/s)
```

**填充到 modelV2** (`fill_model_msg.py:86-90`):

```python
fill_xyzt(modelV2.position, T_IDXS, *plan[0,:,Plan.POSITION].T,
          *plan_stds[0,:,Plan.POSITION].T)       # 含标准差
fill_xyzt(modelV2.velocity, T_IDXS, *plan[0,:,Plan.VELOCITY].T)
fill_xyzt(modelV2.acceleration, T_IDXS, *plan[0,:,Plan.ACCELERATION].T)
fill_xyzt(modelV2.orientation, T_IDXS, *plan[0,:,Plan.T_FROM_CURRENT_EULER].T)
fill_xyzt(modelV2.orientationRate, T_IDXS, *plan[0,:,Plan.ORIENTATION_RATE].T)
```

#### 5.2.1 position — 未来位置

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `position.t` | List(Float32) | 秒 | T_IDXS，33 个时间采样点 |
| `position.x` | List(Float32) | **米** | 前方距离（相对当前位置） |
| `position.y` | List(Float32) | **米** | 右方偏移（相对当前位置） |
| `position.z` | List(Float32) | **米** | 下方偏移（相对当前位置） |
| `position.{x,y,z}Std` | List(Float32) | 米 | 标准差 |

**语义**: 自我车辆在未来各时间点相对于当前位置的位移。例如 `position.x[16] = 30.0` 表示 T_IDXS[16]=2.5 秒后车辆前进了 30 米。

**下游用途**:
- UI 可视化: 绘制驾驶路径（3D 点投影到屏幕）
- 纵向规划: 插值到 MPC 时间网格作为参考轨迹
- 多项式拟合: 4 次多项式系数发布在 `drivingModelData.path` 中

**多项式路径** (`fill_model_msg.py:93`):
```python
fill_xyz_poly(driving_model_data.path, POLY_PATH_DEGREE,
              *plan[0,:,Plan.POSITION].T)
# → path.{x,y,z}Coefficients: 各 5 个系数（4 次多项式 + 常数项）
```

#### 5.2.2 velocity — 未来速度

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `velocity.x` | List(Float32) | **m/s** | 前向速度 |
| `velocity.y` | List(Float32) | **m/s** | 右向速度 |
| `velocity.z` | List(Float32) | **m/s** | 下向速度 |

**语义**: 自我车辆在未来各时间点的瞬时速度。`velocity.x[0]` 是模型对当前时刻前向车速的估计（可与 carState.vEgo 对比，用于 posenet 有效性检测）。

**下游用途**:
- 纵向规划: 速度参考轨迹
- 动作生成: `get_accel_from_plan()` 从速度序列反推期望加速度
- 异常检测: `velocity.x[0]` 与 vEgo 差异过大时触发 posenetInvalid 事件

#### 5.2.3 acceleration — 未来加速度

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `acceleration.x` | List(Float32) | **m/s²** | 前向加速度 |
| `acceleration.y` | List(Float32) | **m/s²** | 右向加速度 |
| `acceleration.z` | List(Float32) | **m/s²** | 下向加速度 |

**下游用途**:
- 纵向规划: 加速度参考轨迹
- UI 可视化: 实验模式下路径颜色编码（正加速=绿，负加速=红）

#### 5.2.4 orientation — 未来姿态角

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `orientation.x` | List(Float32) | **弧度 (rad)** | roll 变化量 |
| `orientation.y` | List(Float32) | **弧度 (rad)** | pitch 变化量 |
| `orientation.z` | List(Float32) | **弧度 (rad)** | yaw 变化量 |

**语义**: 相对于**当前时刻**姿态的累积欧拉角变化（`T_FROM_CURRENT_EULER`）。不是绝对姿态，而是差值。例如 `orientation.z[i]` 表示从现在到 T_IDXS[i] 秒后，车辆累积偏转了多少弧度。

**下游用途**: `get_curvature_from_plan()` 从 yaw 角（`orientation.z`）提取期望曲率。

#### 5.2.5 orientationRate — 未来角速度

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `orientationRate.x` | List(Float32) | **rad/s** | roll 角速度 |
| `orientationRate.y` | List(Float32) | **rad/s** | pitch 角速度 |
| `orientationRate.z` | List(Float32) | **rad/s** | yaw 角速度（即偏航率） |

**下游用途**: `orientationRate.z[0]`（当前 yaw rate）用于曲率计算公式中的 `psi_rate` 项。

### 5.3 Desire State — 当前意图分布

**维度**: 8 维 softmax 分布（与 4.9 节的 8 种意图类型相同）

**解码方式**: softmax (CCE)

**发布**: `modelV2.meta.desireState`

**与 desirePrediction 的区别**:
- `desireState` 是策略网络输出，反映模型**当前**推断的意图状态
- `desirePrediction` 是视觉网络输出，预测**未来** 4 个时间步的意图分布

**下游用途**: 用于变道状态机判断 (`desire_helper.py`)。当 `desireState[laneChangeLeft] + desireState[laneChangeRight]` 的概率从高降到低于 0.02 时，判定变道完成。

---

## 6. 后处理: Action（控制动作推导）

Action 不是直接的网络输出，而是从策略网络的 plan 中**二次推导**得到的控制指令。推导在 `get_action_from_model()` 函数中完成。

**capnp 定义** (`cereal/log.capnp`):
```capnp
struct Action {
  desiredCurvature @0 :Float32;       # 1/m
  desiredAcceleration @1 :Float32;    # m/s²
  shouldStop @2 :Bool;
}
```

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `action.desiredCurvature` | Float32 | **1/米** | 期望路径曲率（曲率 = 1/转弯半径） |
| `action.desiredAcceleration` | Float32 | **m/s²** | 期望纵向加速度 |
| `action.shouldStop` | Bool | — | 是否应停车 |

### 6.1 曲率计算

```python
# drive_helpers.py:62-65
def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
    psi_target = np.interp(action_t, t_idxs, yaws)      # 在 action_t 处插值 yaw 角
    psi_rate = yaw_rates[0]                               # 当前时刻 yaw rate
    return curv_from_psis(psi_target, psi_rate, vego, action_t)

# drive_helpers.py:57-60
def curv_from_psis(psi_target, psi_rate, vego, action_t):
    vego = np.clip(vego, MIN_SPEED, np.inf)               # 最小 1.0 m/s
    curv_from_psi = psi_target / (vego * action_t)
    return 2 * curv_from_psi - psi_rate / vego
```

**时间参数**: `action_t = lat_delay + DT_MDL`，其中:
- `lat_delay` = `liveDelay.lateralDelay + LAT_SMOOTH_SECONDS`（≈ 0.05 + 0.0 = 0.05s）
- `DT_MDL` = 0.05s（模型周期）
- 典型 action_t ≈ 0.1s

**平滑**: 当 v_ego > 0.3 m/s 时，`LAT_SMOOTH_SECONDS = 0.0`（无平滑）。当 v_ego ≤ 0.3 m/s 时保持前一帧曲率值，避免低速抖动。

**约束** (`clip_curvature()` 在控制层应用):
- 最大曲率: `MAX_CURVATURE = 0.2` 1/m（对应 5m 最小转弯半径）
- 最大横向加速度: `MAX_LATERAL_ACCEL_NO_ROLL = 3.0` m/s²
- 最大横向 jerk: `MAX_LATERAL_JERK = 5.0` m/s³

### 6.2 加速度计算

```python
# drive_helpers.py:42-55
def get_accel_from_plan(speeds, accels, t_idxs, action_t, vEgoStopping=0.05):
    v_now = speeds[0]                                       # 模型估计的当前速度
    a_now = accels[0]                                       # 模型估计的当前加速度
    v_target = np.interp(action_t, t_idxs, speeds)         # 在 action_t 处插值目标速度
    a_target = 2 * (v_target - v_now) / action_t - a_now   # 运动学公式反推加速度
    v_target_1sec = np.interp(action_t + 1.0, t_idxs, speeds)
    should_stop = (v_target < vEgoStopping and v_target_1sec < vEgoStopping)
    return a_target, should_stop
```

**时间参数**: `action_t = long_delay + DT_MDL`，其中:
- `long_delay` = `CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS`（≈ 0.15 + 0.3 = 0.45s）
- 典型 action_t ≈ 0.5s

**平滑**: 一阶低通滤波，时间常数 `LONG_SMOOTH_SECONDS = 0.3s`:
```python
alpha = 1 - exp(-DT_MDL / 0.3)  # ≈ 0.154
a_smoothed = alpha * a_target + (1 - alpha) * prev_accel
```

### 6.3 停车判定

当 `action_t` 和 `action_t + 1s` 处的插值速度均低于 `vEgoStopping`（0.05 m/s）时，`shouldStop = True`。

---

## 7. 策略网络输入详解

### 7.1 features_buffer — 视觉特征历史

| 属性 | 值 |
|------|-----|
| Shape | (1, 25, 512) |
| 类型 | float32 |
| 来源 | 视觉网络 hidden_state 的滑动窗口 |
| 窗口 | 25 帧 × 512 维 |

**时间窗口语义**: 25 帧在 5Hz 有效上下文频率下对应 ~5 秒的历史。`InputQueues` 类维护一个固定大小的滑动窗口:

```python
# 每帧: 左移旧数据，追加新的 512D 特征
q['features_buffer'][:, :-1] = q['features_buffer'][:, 1:]
q['features_buffer'][:, -1:] = new_hidden_state

# 取出时按 20/5=4 的间隔重采样
idxs = np.arange(-1, -25, -4)[::-1]  # 等间隔选取
output = q['features_buffer'][:, idxs]
```

### 7.2 desire_pulse — 驾驶意图脉冲

| 属性 | 值 |
|------|-----|
| Shape | (1, 25, 8) |
| 类型 | float32 |
| 来源 | DesireHelper 状态机 |
| 编码 | 脉冲（仅上升沿为 1，其余为 0） |

**脉冲机制**: 模型内部维护意图的"记忆"，因此只在意图**变化**时发送脉冲信号，避免持续信号导致模型内部状态累积偏差:

```python
# modeld.py:265-267
inputs['desire_pulse'][0] = 0
new_desire = np.where(inputs['desire_pulse'] - self.prev_desire > .99,
                       inputs['desire_pulse'], 0)  # 仅保留上升沿
self.prev_desire[:] = inputs['desire_pulse']
```

**DesireHelper 状态机** (`desire_helper.py`):

```
  转向灯开启                方向盘施力              变道概率<2%
off ─────→ preLaneChange ──────→ laneChangeStarting ──────→ laneChangeFinishing
  │                  │                                              │
  └──── 灯灭/低速 ───┘                                              │
                                              概率>99% ────────────→ off
```

### 7.3 traffic_convention — 交通规则

| 属性 | 值 |
|------|-----|
| Shape | (1, 2) |
| 类型 | float32 |
| 编码 | 独热: [LHD, RHD] |

**语义**: `[1, 0]` = 左侧通行（Left-Hand Drive，如中国、美国），`[0, 1]` = 右侧通行（Right-Hand Drive，如英国、日本）。由 driverMonitoringState.isRHD 决定。

---

## 8. Confidence 输出（置信度评级）

Confidence 不是网络直接输出，而是从 meta 的脱离概率推导的综合评分。

| 值 | 含义 | 阈值条件 |
|----|------|---------|
| `green` | 正常 | score < 0.01165 |
| `yellow` | 中等 | 0.01165 ≤ score < 0.06157 |
| `red` | 低置信度 | score ≥ 0.06157 |

**Score 计算** (`fill_model_msg.py:151-166`):

```python
# 1. 计算综合脱离概率（任一脱离事件发生的概率）
any_disengage = 1 - (1-brake_disengage) * (1-gas_disengage) * (1-steer_override)

# 2. 转换为独立条件概率（每 2s 区间）
ind_disengage[0] = any_disengage[0]
ind_disengage[i] = diff(any_disengage[i]) / (1 - any_disengage[i-1])

# 3. 滚动缓冲区（5 帧 × 5 个时间区间），每 2 秒更新一次
disengage_buffer[:-5] = disengage_buffer[5:]
disengage_buffer[-5:] = ind_disengage

# 4. 对角线平均（buffer[0][4], buffer[1][3], buffer[2][2], buffer[3][1], buffer[4][0]）
score = mean(disengage_buffer[i*5 + 4-i] for i in range(5))
```

**对角线取值的含义**: 第 i 个历史帧预测 (4-i) 个时间步后的脱离概率。这些预测指向大致相同的未来时刻，取平均提供更稳定的评估。

---

## 9. 输出解码方法

### 9.1 MDN (Mixture Density Network)

适用于连续量（位置、速度等）。网络输出 2N 维向量:
- **前 N 维 (μ)**: 均值，直接就是物理量数值
- **后 N 维 (log σ)**: 对数标准差，经 `exp(clip(x, -∞, 11))` 转换为标准差

```python
# parse_model_outputs.py:50-52
n_values = (raw.shape[2] - out_N) // 2
pred_mu = raw[:, :, :n_values]
pred_std = safe_exp(raw[:, :, n_values:2*n_values])  # exp(clip(x, -inf, 11))
```

### 9.2 MHP (Multi-Hypothesis Prediction)

对于 plan 和 lead，网络生成多个假设并附带权重:

- **Plan**: 5 个假设（`PLAN_MHP_N=5`），每个假设独立预测 33 时间步 × 15 维。通过 softmax 权重选择最优的 **1 个**假设 (`PLAN_MHP_SELECTION=1`)。
- **Lead**: 2 个假设（`LEAD_MHP_N=2`），选择权重最高的 **3 个**目标 (`LEAD_MHP_SELECTION=3`)。

**选择机制**:
```python
# parse_model_outputs.py:54-76
weights = softmax(raw[:, :, -out_N:])          # 假设权重
idxs = np.argsort(weights[:, :, 0])[::-1]     # 按权重排序
pred_mu_final = pred_mu[idxs[0]]              # 取最高权重假设
```

### 9.3 Sigmoid (BCE)

适用于独立事件概率（meta、lane_lines_prob、lead_prob）。将原始 logit 通过 `1/(1+exp(-x))` 映射到 (0, 1)。

### 9.4 Softmax (CCE)

适用于互斥分类分布（desire_state、desire_pred）。在最后一维做归一化:

```python
# parse_model_outputs.py:11-18
def softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)  # 数值稳定
    x = safe_exp(x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x
```

---

## 10. 消息发布总览

模型每帧发布 3 条 cereal 消息:

### 10.1 modelV2 — 完整模型输出

包含策略网络 + 视觉网络的全部结构化输出。供 UI 可视化、纵向/横向规划、radard 等使用。

```capnp
struct ModelDataV2 {
  frameId, frameIdExtra, frameAge, frameDropPerc, timestampEof,
  modelExecutionTime, rawPredictions,
  position, orientation, velocity, orientationRate, acceleration,  # Plan 输出
  laneLines[4], laneLineProbs, laneLineStds,                      # 车道线
  roadEdges[2], roadEdgeStds,                                      # 路沿
  leadsV3[3],                                                      # 前车
  meta,                                                            # 元事件
  confidence,                                                      # 置信度评级
  action                                                           # 控制动作
}
```

### 10.2 drivingModelData — 简化控制数据

仅包含控制层所需的最小数据集，减少序列化/反序列化开销。

```capnp
struct DrivingModelData {
  frameId, frameIdExtra, frameDropPerc, modelExecutionTime,
  action,         # Action{desiredCurvature, desiredAcceleration, shouldStop}
  laneLineMeta,   # LaneLineMeta{leftY, rightY, leftProb, rightProb}
  meta,           # MetaData{laneChangeState, laneChangeDirection}
  path            # PolyPath{xCoefficients, yCoefficients, zCoefficients}
}
```

### 10.3 cameraOdometry — 相机自运动

独立于 modelV2 发布的视觉里程计数据。

```capnp
struct CameraOdometry {
  frameId, timestampEof,
  trans[3], rot[3], transStd[3], rotStd[3],           # 6D 自运动
  wideFromDeviceEuler[3], wideFromDeviceEulerStd[3],   # 广角相机对齐
  roadTransformTrans[3], roadTransformTransStd[3]       # 路面变换
}
```

---

## 11. 图像预处理流水线

### 11.1 总览

```
VisionIPC (NV12/YUV420)
    ↓
[OpenCL: warpPerspective]  ← warp_matrix (在线标定)
    ↓
[OpenCL: loadyuv]          ← Y 平面 4:1 重排 + UV 分离
    ↓
uint8 张量 (1, 12, 128, 256)
    ↓
[Tinygrad 推理]
    ↓
float32 输出向量
```

### 11.2 时间帧堆叠

每次推理输入 2 帧图像（`N_FRAMES = 2`），间隔 `temporal_skip = MODEL_RUN_FREQ / MODEL_CONTEXT_FREQ - 1 = 3` 帧（在 20Hz 中间隔 3 帧 = 200ms）:

```
20Hz 帧序列:  ... f₁ f₂ f₃ f₄ f₅ f₆ f₇ f₈ ...
                   ↑              ↑
                 frame[t-3]    frame[t]     → 拼接为 12 通道输入
```

- 每帧 6 通道（Y 平面 4:1 重排后 4 通道 + U 1 通道 + V 1 通道）
- 2 帧 × 6 通道 = 12 通道
- 分辨率 128 × 256（模型内部分辨率，warp 变换后）

### 11.3 双相机输入

| 输入名称 | 源相机 | 模型坐标系 | warp 矩阵 |
|---------|--------|-----------|-----------|
| `img` | 主路 (fcam/ecam) | MEDModel | `model_transform_main` |
| `big_img` | 广角 (ecam) | SBIGModel | `model_transform_extra` |

- **双相机模式** (comma 3X): `img` = fcam, `big_img` = ecam
- **单广角模式**: `img` = ecam (MEDModel warp), `big_img` = ecam (SBIGModel warp)
- 命名中 `big_img` 对应 SBIGModel（S=Small size, BIG=wide FOV）

---

## 12. 关键常量速查

| 常量 | 值 | 定义位置 | 含义 |
|------|-----|---------|------|
| `IDX_N` | 33 | constants.py | 采样点数（时间/空间） |
| `T_IDXS` | [0..10] s | constants.py | 时间采样网格 |
| `X_IDXS` | [0..192] m | constants.py | 距离采样网格 |
| `LEAD_T_IDXS` | [0,2,4,6,8,10] s | constants.py | 前车时间采样 |
| `META_T_IDXS` | [2,4,6,8,10] s | constants.py | 脱离预测时间点 |
| `NUM_LANE_LINES` | 4 | constants.py | 车道线数量 |
| `NUM_ROAD_EDGES` | 2 | constants.py | 路沿数量 |
| `PLAN_WIDTH` | 15 | constants.py | plan 每时间点维度 |
| `LEAD_WIDTH` | 4 | constants.py | lead 每时间点维度 (x,y,v,a) |
| `LANE_LINES_WIDTH` | 2 | constants.py | 车道线每采样点维度 (y,z) |
| `POSE_WIDTH` | 6 | constants.py | 位姿维度 (3 平移 + 3 旋转) |
| `FEATURE_LEN` | 512 | constants.py | 隐藏状态维度 |
| `DESIRE_LEN` | 8 | constants.py | 意图类别数 |
| `PLAN_MHP_N` | 5 | constants.py | plan 假设数 |
| `PLAN_MHP_SELECTION` | 1 | constants.py | plan 选取数 |
| `LEAD_MHP_N` | 2 | constants.py | lead 假设数 |
| `LEAD_MHP_SELECTION` | 3 | constants.py | lead 选取数 |
| `MODEL_RUN_FREQ` | 20 Hz | constants.py | 模型推理频率 |
| `MODEL_CONTEXT_FREQ` | 5 Hz | constants.py | 上下文帧率 |
| `N_FRAMES` | 2 | constants.py | 时间帧堆叠数 |
| `DT_MDL` | 0.05 s | realtime.py | 模型周期 (1/20Hz) |
| `HEIGHT_INIT` | 1.22 m | calibrationd.py | 相机离路面高度基线值 |
| `CONFIDENCE_BUFFER_LEN` | 5 | constants.py | 置信度滚动窗口长度 |
| `RYG_GREEN` | 0.01165 | constants.py | 置信度绿色阈值 |
| `RYG_YELLOW` | 0.06157 | constants.py | 置信度黄色阈值 |

---

## 13. 视觉网络 vs 策略网络输出对比

| 维度 | 视觉网络 (1576D) | 策略网络 (1000D) |
|------|------------------|------------------|
| **环境几何** | 车道线 (4×33×2), 路沿 (2×33×2) | — |
| **目标检测** | 前车 (3×6×4), lead_prob (3) | — |
| **自运动** | pose (6D), wide_from_device (3D), road_transform (6D) | — |
| **事件概率** | meta (55D), desire_pred (32D), lane_lines_prob (8D) | — |
| **特征传递** | hidden_state (512D) → 策略网络 | — |
| **轨迹规划** | — | plan (33×15), 含 position/velocity/acceleration/orientation |
| **意图状态** | — | desire_state (8D) |
| **控制动作** | — | → 后处理: Action{curvature, accel, stop} |

**设计哲学**: 视觉网络负责"看到了什么"（感知），策略网络负责"应该怎么做"（规划）。隐藏状态作为唯一桥梁，迫使视觉网络学会将图像信息压缩为 512 维的高效表征。
