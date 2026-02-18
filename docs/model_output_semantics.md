# supercombo 模型输出数据语义参考

本文档详细描述 openpilot supercombo 模型（经 `modelV2` 消息发布）的每个输出字段的物理语义、数值单位和坐标系约定。

**相关源码**:
- 消息定义: `cereal/log.capnp` (ModelDataV2)
- 输出填充: `selfdrive/modeld/fill_model_msg.py`
- 输出解析: `selfdrive/modeld/parse_model_outputs.py`
- 常量定义: `selfdrive/modeld/constants.py`

---

## 1. 坐标系

模型的所有 3D 输出（位置、速度、加速度、车道线、路沿、前车）都在**标定坐标系 (Calibrated Frame)** 中表述。

### 1.1 标定坐标系定义

标定坐标系是消除了相机安装偏差后的参考系。它与车辆坐标系 (Car Frame) 在 pitch 和 yaw 上对齐，与设备坐标系 (Device Frame) 在 roll 上对齐，原点位于设备（相机）处。

**轴定义**（参见 `common/transformations/README.md`）:

| 轴 | 方向 | 正值含义 |
|----|------|---------|
| **x** | 前方 (Forward) | 离车越远，值越大 |
| **y** | 右方 (Right) | 车辆右侧为正 |
| **z** | 下方 (Down) | 路面方向为正 |

> **注意**: capnp schema 注释写着 "All SI units and in device frame"，但 Device Frame 和 Calibrated Frame 都是 [Forward, Right, Down]，仅通过 rpyCalib 旋转对齐。模型的 warp 矩阵已经将图像变换到标定坐标系，因此模型输出本质上在标定坐标系中。

### 1.2 相关坐标系对比

| 坐标系 | x | y | z | 用途 |
|--------|---|---|---|------|
| Device | 前 | 右 | 下 | 物理传感器安装 |
| Calibrated | 前 | 右 | 下 | 模型输出、控制计算 |
| View | 右 | 下 | 前 | 图像投影中间坐标 |
| Car | 前 | 右 | 下 | 与路面和车辆方向对齐 |

### 1.3 投影关系

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

---

## 2. 采样网格

模型输出不是对均匀网格的预测，而是在二次函数分布的采样点上。

### 2.1 时间索引 T_IDXS

用于 plan 相关输出（position, velocity, acceleration, orientation, orientationRate）。

```python
T_IDXS[i] = 10.0 × (i/32)²    # i = 0, 1, ..., 32
```

33 个时间点，覆盖 **0 到 10 秒**，近处密集、远处稀疏:

| 索引 | 0 | 1 | 2 | 5 | 8 | 16 | 24 | 32 |
|------|---|---|---|---|---|----|----|-----|
| 时间(s) | 0.0 | 0.010 | 0.039 | 0.244 | 0.625 | 2.5 | 5.625 | 10.0 |

**物理意义**: 预测自我车辆未来 10 秒的状态。近处采样密（~10ms 间隔），远处采样疏（~1.2s 间隔），反映了对近期精确性和远期趋势的不同需求。

### 2.2 距离索引 X_IDXS

用于车道线 (laneLines) 和路沿 (roadEdges) 的空间采样。

```python
X_IDXS[i] = 192.0 × (i/32)²    # i = 0, 1, ..., 32
```

33 个距离点，覆盖 **0 到 192 米**:

| 索引 | 0 | 1 | 2 | 5 | 8 | 16 | 24 | 32 |
|------|---|---|---|---|---|----|----|-----|
| 距离(m) | 0.0 | 0.188 | 0.75 | 4.69 | 12.0 | 48.0 | 108.0 | 192.0 |

**物理意义**: 在不同前方距离处采样车道线和路沿的横向偏移和高度。近处密采保证近处几何精度，远处稀疏减少不确定性累积。

### 2.3 前车时间索引 LEAD_T_IDXS

用于前车预测 (leadsV3)。

```python
LEAD_T_IDXS = [0., 2., 4., 6., 8., 10.]    # 6 个时间点
```

均匀分布，每 2 秒一个采样点，共 **6 个点**覆盖 0-10 秒。

### 2.4 元事件时间索引 META_T_IDXS

用于脱离预测 (disengagePredictions)。

```python
META_T_IDXS = [2., 4., 6., 8., 10.]    # 5 个时间点
```

---

## 3. Plan 输出（自我轨迹预测）

Plan 是策略网络的核心输出，预测自我车辆未来 10 秒的完整运动状态。

**解码方式**: MDN + MHP（5 个假设取最优 1 个）

### 3.1 position — 未来位置

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `position.t` | List(Float32) | 秒 | T_IDXS，33 个时间采样点 |
| `position.x` | List(Float32) | **米** | 前方距离（相对当前位置） |
| `position.y` | List(Float32) | **米** | 右方偏移（相对当前位置） |
| `position.z` | List(Float32) | **米** | 下方偏移（相对当前位置） |
| `position.xStd` | List(Float32) | 米 | x 标准差 |
| `position.yStd` | List(Float32) | 米 | y 标准差 |
| `position.zStd` | List(Float32) | 米 | z 标准差 |

**语义**: 自我车辆在未来各时间点相对于当前位置的位移。例如 `position.x[16] = 30.0` 表示 T_IDXS[16]=2.5 秒后车辆前进了 30 米。

**下游用途**:
- UI 可视化: 绘制驾驶路径（3D 点投影到屏幕）
- 纵向规划: 插值到 MPC 时间网格作为参考轨迹
- 多项式拟合: 4 次多项式系数发布在 `drivingModelData.path` 中

### 3.2 velocity — 未来速度

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

### 3.3 acceleration — 未来加速度

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `acceleration.x` | List(Float32) | **m/s²** | 前向加速度 |
| `acceleration.y` | List(Float32) | **m/s²** | 右向加速度 |
| `acceleration.z` | List(Float32) | **m/s²** | 下向加速度 |

**下游用途**:
- 纵向规划: 加速度参考轨迹
- UI 可视化: 实验模式下路径颜色编码（正加速=绿，负加速=红）

### 3.4 orientation — 未来姿态角

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `orientation.x` | List(Float32) | **弧度 (rad)** | roll 变化量 |
| `orientation.y` | List(Float32) | **弧度 (rad)** | pitch 变化量 |
| `orientation.z` | List(Float32) | **弧度 (rad)** | yaw 变化量 |

**语义**: 相对于**当前时刻**姿态的累积欧拉角变化（`T_FROM_CURRENT_EULER`）。不是绝对姿态，而是差值。例如 `orientation.z[i]` 表示从现在到 T_IDXS[i] 秒后，车辆累积偏转了多少弧度。

**下游用途**: `get_curvature_from_plan()` 从 yaw 角（`orientation.z`）提取期望曲率。

### 3.5 orientationRate — 未来角速度

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `orientationRate.x` | List(Float32) | **rad/s** | roll 角速度 |
| `orientationRate.y` | List(Float32) | **rad/s** | pitch 角速度 |
| `orientationRate.z` | List(Float32) | **rad/s** | yaw 角速度（即偏航率） |

**下游用途**: `orientationRate.z[0]`（当前 yaw rate）用于曲率计算公式中的 `psi_rate` 项。

---

## 4. 车道线输出 (laneLines)

模型预测 **4 条车道线**，编号 0-3:
- `laneLines[0]`: 最左侧车道线（远左）
- `laneLines[1]`: 左相邻车道线
- `laneLines[2]`: 右相邻车道线
- `laneLines[3]`: 最右侧车道线（远右）

**解码方式**: MDN（均值 + 标准差）

### 4.1 几何数据

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

**重要**: 在 `fill_model_msg.py` 第 105 行可以看到填充逻辑:
```python
fill_xyzt(lane_line, LINE_T_IDXS,
          np.array(ModelConstants.X_IDXS),       # x = 固定采样距离
          net_output_data['lane_lines'][0,i,:,0], # y = 模型预测的横向偏移
          net_output_data['lane_lines'][0,i,:,1]) # z = 模型预测的垂直偏移
```

### 4.2 概率和标准差

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `laneLineProbs` | List(Float32) | **概率 (0~1)** | 4 条车道线的存在概率 |
| `laneLineStds` | List(Float32) | **米** | 4 条车道线的位置标准差 |

**laneLineProbs 的生成**: 网络输出 8 个 sigmoid 值（`lane_lines_prob`），取奇数索引 `[1, 3, 5, 7]` 得到 4 条线的概率。

**下游用途**:
- LDW (车道偏离预警): 当 `laneLineProbs[1] > 0.5` 且 `laneLines[1].y[0]` 距离小于阈值时触发
- UI 可视化: 车道线绘制透明度 = `clip(prob, 0, 0.7)`

---

## 5. 路沿输出 (roadEdges)

模型预测 **2 条路沿线**:
- `roadEdges[0]`: 左侧路沿
- `roadEdges[1]`: 右侧路沿

**解码方式**: MDN

### 5.1 几何数据

结构与车道线完全相同:

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `roadEdges[i].x` | List(Float32) | **米** | 固定为 X_IDXS（33 个距离点） |
| `roadEdges[i].y` | List(Float32) | **米** | 横向偏移（右方为正） |
| `roadEdges[i].z` | List(Float32) | **米** | 垂直偏移（下方为正） |

### 5.2 标准差

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `roadEdgeStds` | List(Float32) | **米** | 2 条路沿的位置标准差 |

**下游用途**: UI 绘制透明度 = `clip(1.0 - std, 0, 1.0)`，标准差越小越不透明（越确定）。

---

## 6. 前车输出 (leadsV3)

模型预测 **3 个前车目标**，按相关性排序。每个前车包含未来 10 秒的状态时间序列。

**解码方式**: MDN + MHP（2 个假设取 3 个目标）

### 6.1 数据结构

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `leadsV3[i].prob` | Float32 | **概率 (0~1)** | 该前车存在的概率 |
| `leadsV3[i].probTime` | Float32 | **秒** | 概率对应的时间偏移（0/2/4 秒） |
| `leadsV3[i].t` | List(Float32) | 秒 | LEAD_T_IDXS = [0, 2, 4, 6, 8, 10]，6 个时间点 |
| `leadsV3[i].x` | List(Float32) | **米** | 前向距离（6 个时间点） |
| `leadsV3[i].xStd` | List(Float32) | 米 | x 标准差 |
| `leadsV3[i].y` | List(Float32) | **米** | 横向偏移（6 个时间点） |
| `leadsV3[i].yStd` | List(Float32) | 米 | y 标准差 |
| `leadsV3[i].v` | List(Float32) | **m/s** | 绝对法向速度（6 个时间点） |
| `leadsV3[i].vStd` | List(Float32) | m/s | v 标准差 |
| `leadsV3[i].a` | List(Float32) | **m/s²** | 速度的导数（6 个时间点） |
| `leadsV3[i].aStd` | List(Float32) | m/s² | a 标准差 |

### 6.2 字段语义详解

**x（前向距离）**: 前车相对于自我车辆的纵向距离。`leadsV3[0].x[0]` 是当前时刻最近前车的距离。正值表示前方。

**y（横向偏移）**: 前车相对于自我车辆的横向位置。capnp schema 注释为 "x and y are relative position in device frame"（设备坐标系下 y=右方为正）。

**v（绝对速度）**: capnp schema 注释为 "v absolute norm speed"，是前车的**绝对**速度标量，不是相对速度。下游 radard 通过 `v_rel = lead.v[0] - model_v_ego` 计算相对速度。

**a（加速度）**: v 的时间导数，即前车的纵向加速度。

**probTime**: 3 个前车目标的概率分别对应不同的时间偏移（`LEAD_T_OFFSETS = [0, 2, 4]` 秒）。`leadsV3[0].probTime = 0` 表示当前时刻的前车，`leadsV3[1].probTime = 2` 表示 2 秒后的前车（可能是不同的车辆）。

**下游用途**:
- radard: 视觉-雷达融合，当 `prob > 0.5` 时视为有效前车
- 纵向规划: 前车距离和速度是 ACC 的关键输入
- UI 可视化: 绘制 chevron 标记，显示距离标签

---

## 7. Action 输出（控制动作）

Action 不是直接的网络输出，而是从 plan 中**二次推导**得到的控制指令。

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `action.desiredCurvature` | Float32 | **1/米** | 期望路径曲率（曲率 = 1/转弯半径） |
| `action.desiredAcceleration` | Float32 | **m/s²** | 期望纵向加速度 |
| `action.shouldStop` | Bool | — | 是否应停车 |

### 7.1 曲率计算

```python
# selfdrive/controls/lib/drive_helpers.py
psi_target = interp(action_t, T_IDXS, orientation.z)       # yaw 角插值
curvature = 2 * psi_target / (v_ego * action_t) - yaw_rate / v_ego
```

其中 `action_t = lateral_delay + DT_MDL`，包含了横向执行器延迟补偿。

### 7.2 加速度计算

```python
v_target = interp(action_t, T_IDXS, velocity.x)
a_target = 2 * (v_target - v_ego) / action_t - acceleration.x[0]
```

经过一阶低通滤波平滑（`LONG_SMOOTH_SECONDS = 0.3s`）。

### 7.3 停车判定

当 `action_t` 和 `action_t + 1s` 处的插值速度均低于 `vEgoStopping`（0.05 m/s）时，`shouldStop = True`。

---

## 8. Meta 输出（元事件预测）

Meta 是一个 55 维的 sigmoid 向量，编码了丰富的驾驶事件概率。

### 8.1 消息结构

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `meta.engagedProb` | Float32 | **概率 (0~1)** | 驾驶员接管/参与概率 |
| `meta.desireState` | List(Float32) | **概率分布** | 当前意图状态（8 维 softmax） |
| `meta.desirePrediction` | List(Float32) | **概率分布** | 未来意图预测（4×8=32 维） |
| `meta.hardBrakePredicted` | Bool | — | 前碰撞预警（FCW）标志 |
| `meta.laneChangeState` | Enum | — | 变道状态（off/pre/starting/finishing） |
| `meta.laneChangeDirection` | Enum | — | 变道方向（none/left/right） |

### 8.2 脱离预测 (disengagePredictions)

预测未来 2/4/6/8/10 秒内各类脱离事件的**累积概率**:

| 字段 | 时间点 | 单位 | 描述 |
|------|--------|------|------|
| `brakeDisengageProbs` | [2,4,6,8,10]s | 概率 (0~1) | 制动导致脱离 |
| `gasDisengageProbs` | [2,4,6,8,10]s | 概率 (0~1) | 油门导致脱离 |
| `steerOverrideProbs` | [2,4,6,8,10]s | 概率 (0~1) | 方向盘接管 |
| `brake3MetersPerSecondSquaredProbs` | [2,4,6,8,10]s | 概率 (0~1) | ≥3 m/s² 急刹 |
| `brake4MetersPerSecondSquaredProbs` | [2,4,6,8,10]s | 概率 (0~1) | ≥4 m/s² 急刹 |
| `brake5MetersPerSecondSquaredProbs` | [2,4,6,8,10]s | 概率 (0~1) | ≥5 m/s² 急刹 |
| `gasPressProbs` | [2,4,6,8,10]s | 概率 (0~1) | 油门踏板按压 |
| `brakePressProbs` | [2,4,6,8,10]s | 概率 (0~1) | 制动踏板按压 |

### 8.3 Desire 意图的 8 种类型

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

---

## 9. Confidence 输出（置信度评级）

| 值 | 含义 | 阈值条件 |
|----|------|---------|
| `green` | 正常 | score < 0.01165 |
| `yellow` | 中等 | 0.01165 ≤ score < 0.06157 |
| `red` | 低置信度 | score ≥ 0.06157 |

Score 基于滚动 5 帧窗口的综合脱离概率对角线平均值。

---

## 10. Camera Odometry 输出（相机里程计）

由视觉网络直接输出（不经过策略网络），发布在独立的 `cameraOdometry` 消息中。

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| `trans` | List(Float32) × 3 | **m/s** | 平移速度 [tx, ty, tz]（设备坐标系） |
| `rot` | List(Float32) × 3 | **rad/s** | 旋转角速度 [roll, pitch, yaw]（设备坐标系） |
| `transStd` | List(Float32) × 3 | m/s | 平移标准差 |
| `rotStd` | List(Float32) × 3 | rad/s | 旋转标准差 |
| `wideFromDeviceEuler` | List(Float32) × 3 | **弧度 (rad)** | 广角相机相对设备的欧拉角 [roll, pitch, yaw] |
| `roadTransformTrans` | List(Float32) × 3 | **m/s** | 路面坐标变换的平移分量 |

**trans 的语义**: 是速度而非位移。`trans[0]` ≈ 前向车速，`trans[2]` 与相机高度相关。

**下游用途**:
- locationd: 输入 EKF 传感器融合，与 IMU/GPS 联合估计车辆状态
- calibrationd: 从 trans（需 vEgo > 阈值）估计相机标定参数；从 roadTransformTrans[2] 估计相机高度

---

## 11. 网络输出解码方法

### 11.1 MDN (Mixture Density Network)

适用于连续量（位置、速度等）。网络输出包含:
- **均值 (μ)**: 直接就是物理量数值
- **对数标准差 (log σ)**: 经 `exp(clip(x, -∞, 11))` 转换为标准差

### 11.2 MHP (Multi-Hypothesis Prediction)

对于 plan 和 lead，网络生成多个假设:
- **Plan**: 5 个假设（`PLAN_MHP_N=5`），通过 softmax 权重选择最优的 **1 个**
- **Lead**: 2 个假设（`LEAD_MHP_N=2`），选择权重最高的 **3 个**目标

### 11.3 Sigmoid

适用于独立事件概率。将原始 logit 通过 `1/(1+exp(-x))` 映射到 (0, 1)。

### 11.4 Softmax

适用于互斥分类分布（desire 意图等）。在最后一维做归一化。

---

## 12. 关键常量速查

| 常量 | 值 | 定义位置 | 含义 |
|------|-----|---------|------|
| `IDX_N` | 33 | constants.py | 采样点数（时间/空间） |
| `T_IDXS` | [0..10] s | constants.py | 时间采样网格 |
| `X_IDXS` | [0..192] m | constants.py | 距离采样网格 |
| `LEAD_T_IDXS` | [0,2,4,6,8,10] s | constants.py | 前车时间采样 |
| `NUM_LANE_LINES` | 4 | constants.py | 车道线数量 |
| `NUM_ROAD_EDGES` | 2 | constants.py | 路沿数量 |
| `PLAN_WIDTH` | 15 | constants.py | plan 每时间点维度 |
| `LEAD_WIDTH` | 4 | constants.py | lead 每时间点维度 (x,y,v,a) |
| `LANE_LINES_WIDTH` | 2 | constants.py | 车道线每采样点维度 (y,z) |
| `PLAN_MHP_N` | 5 | constants.py | plan 假设数 |
| `LEAD_MHP_N` | 2 | constants.py | lead 假设数 |
| `MODEL_RUN_FREQ` | 20 Hz | constants.py | 模型推理频率 |
| `HEIGHT_INIT` | 1.22 m | calibrationd.py | 相机离路面高度基线值 |
