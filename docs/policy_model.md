# openpilot 策略模型（Policy Model）深度解析

> 本文档深度剖析 openpilot 驾驶模型中策略模型的完整数据流：输入张量语义、输出张量语义、数据来源与下游消费，以及与上下游系统的集成关系。

---

## 一、整体架构

openpilot 采用 **视觉-策略两阶段架构**，两个神经网络串联运行：

```
┌─────────────────────────────────────────────────────────────────┐
│                         Camera Layer (camerad)                   │
│  主相机 (road) 512×256 YUV420                                    │
│  广角相机 (wide road) 512×256 YUV420          (20 Hz)            │
└───────────────────────┬─────────────────────────────────────────┘
                        │ VisionIPC
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Vision Network (20 Hz)                       │
│  输入：img (1,12,128,256)  +  big_img (1,12,128,256)             │
│  输出：1576维原始向量                                             │
│    ├─ 感知输出（车道线、前车、相机运动）                           │
│    └─ hidden_state [1064:1576] → 512维特征                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │ hidden_state 入队（25帧滑动窗口）
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Policy Network (20 Hz)                       │
│  输入：features_buffer (1,25,512)                                │
│         desire_pulse   (1,25,8)                                  │
│         traffic_convention (1,2)                                 │
│  输出：1000维原始向量                                             │
│    ├─ plan [0:990] → 未来10秒行驶规划                            │
│    └─ desire_state [990:998] → 当前意图分布                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │ 动作生成
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│               Action & Control Layer                             │
│  desiredCurvature  → controlsd → 横向转向                        │
│  desiredAcceleration → plannerd → 纵向加速/制动                   │
│  cameraOdometry    → locationd → 传感器融合/定位                  │
└─────────────────────────────────────────────────────────────────┘
```

代码入口：`selfdrive/modeld/modeld.py`

---

## 二、视觉网络输入

### 2.1 图像张量

| 张量名 | 形状 | 数据类型 | 含义 |
|--------|------|----------|------|
| `img` | `(1, 12, 128, 256)` | uint8 | 主相机（道路正前方） |
| `big_img` | `(1, 12, 128, 256)` | uint8 | 广角相机（近距/侧向） |

**12通道编码格式（YUV420 双帧）**

```
12通道 = 2帧 × 6通道/帧
每帧6通道：
  ch0: Y_00（Y通道，左上2×2块）
  ch1: Y_01（Y通道，右上2×2块）
  ch2: Y_10（Y通道，左下2×2块）
  ch3: Y_11（Y通道，右下2×2块）
  ch4: U（色度U，1/4分辨率）
  ch5: V（色度V，1/4分辨率）

2帧时间间隔：temporal_skip = 3 → 约0.2秒（在20Hz下相当于4帧间隔）
```

### 2.2 图像预处理流水线

相机原始帧在送入网络前经过以下 GPU（OpenCL）处理：

**步骤1：透视变换（warpPerspective）**

将物理相机坐标系变换到模型虚拟相机坐标系：

```
warp = model_intrinsics @ view_from_device
              @ device_from_calib @ calib_from_model

其中：
  calib_from_model = 从模型虚拟内参到标定坐标的变换
  device_from_calib = 依赖 rpyCalib（在线标定欧拉角）
  view_from_device  = 从设备坐标到视图坐标的固定旋转
  model_intrinsics  = 虚拟相机内参（固定值）
```

`rpyCalib` 由 `liveCalibration` 消息实时更新，实现在线标定补偿。

**步骤2：YUV重排（loadYUV）**

将 H×W YUV420 平面格式拆解为 6通道张量，通过 OpenCL 核函数执行。

**硬件优化（TICI/comma 3X）：**
- OpenCL 缓冲直接映射为 tinygrad 张量（零拷贝）
- 避免 GPU→CPU→GPU 往返，延迟约 50ms/帧

---

## 三、视觉网络输出（1576 维）

视觉网络原始输出为 1576 维向量，按以下语义切片解析：

```
[0    : 55  ] meta              55维  脱离/事件预测
[55   : 87  ] desire_pred       32维  未来意图分布（4×8）
[87   : 99  ] pose              12维  相机6DOF位姿 + 标准差
[99   : 105 ] wide_from_device_euler  6维  广角相机欧拉角 + 标准差
[105  : 117 ] road_transform    12维  路面坐标变换 + 标准差
[117  : 645 ] lane_lines       528维  4条车道线（MDN）
[645  : 653 ] lane_lines_prob    8维  4条车道线存在概率
[653  : 917 ] road_edges       264维  2条路沿（MDN）
[917  : 1061] lead             144维  前车预测（MHP+MDN）
[1061 : 1064] lead_prob          3维  3个前车目标存在概率
[1064 : 1576] hidden_state     512维  ← 传给策略网络的特征向量
```

### 3.1 车道线（lane_lines，528维）

**MDN（混合密度网络）格式**：

```python
# 全均值在前、全sigma在后
raw = lane_lines_raw  # shape: (528,)
# reshape → (4条线, 33点, 4维: y_mean, z_mean, y_std, z_std)
n = 4 * 33 * 2  # = 264（均值部分）
means = raw[:n].reshape(4, 33, 2)   # [y, z] 坐标均值
stds  = exp(raw[n:].reshape(4, 33, 2))  # [y, z] 标准差
```

- 4条线：左左、左、右、右右（从左到右）
- 33个时间/空间点：对应 `T_IDXS` 时间索引
- 坐标系：标定坐标系（x前、y左、z上）

### 3.2 前车预测（lead，144维）

**MHP（多假设预测）+ MDN 格式**：

```python
# 2个假设，每假设6时间点×(4维均值+4维std+1维权重)
# lead_t = [0, 2, 4, 6, 8, 10] 秒
# 每个时间点：[x, y, v, a]（前向距离、横向偏移、相对速度、相对加速度）
# 最终选择权重最高的3个目标输出
```

### 3.3 相机运动（pose，12维）

```python
pose = raw[87:99]
# [tx, ty, tz, rx, ry, rz]   均值（6维）
# [tx_std, ty_std, tz_std, rx_std, ry_std, rz_std]  标准差（6维）
# 单位：平移 m/s，旋转 rad/s
# 用途：发布为 cameraOdometry → locationd 传感器融合
```

### 3.4 Meta 事件预测（55维）

| 切片索引 | 含义 | 时间点 |
|----------|------|--------|
| `0:1` | ENGAGED 接管概率 | 当前 |
| `1,7,13,19,25` | GAS_DISENGAGE 油门脱离 | 2/4/6/8/10s |
| `2,8,14,20,26` | BRAKE_DISENGAGE 制动脱离 | 2/4/6/8/10s |
| `3,9,15,21,27` | STEER_OVERRIDE 方向盘接管 | 2/4/6/8/10s |
| `4,10,16,22,28` | HARD_BRAKE_3 急刹（3m/s²） | 2/4/6/8/10s |
| `5,11,17,23,29` | HARD_BRAKE_4 急刹（4m/s²） | 2/4/6/8/10s |
| `6,12,18,24,30` | HARD_BRAKE_5 急刹（5m/s²） | 2/4/6/8/10s |
| `31,35,39,43,47,51` | GAS_PRESS 油门踩踏 | 0/2/4/6/8/10s |
| `32,36,40,44,48,52` | BRAKE_PRESS 制动踩踏 | 0/2/4/6/8/10s |
| `33,37,41,45,49,53` | LEFT_BLINKER 左转灯 | 0/2/4/6/8/10s |
| `34,38,42,46,50,54` | RIGHT_BLINKER 右转灯 | 0/2/4/6/8/10s |

### 3.5 desire_pred（未来意图分布，32维）

| 属性 | 值 |
|------|-----|
| 切片位置 | `[55:87]` |
| 解析方式 | `parse_categorical_crossentropy` → softmax |
| 形状（解析后） | `(4, 8)`：4个时间步 × 8种意图概率 |
| 发布字段 | `modelV2.meta.desirePrediction`（展平为32维列表） |

`desire_pred` 是视觉网络对**未来短时间内车辆变道意图**的概率预测，完全基于视觉输入（摄像头图像），不依赖驾驶员是否拨动转向灯。

#### 下游消费：LDW 车道偏离预警（`selfdrive/controls/lib/ldw.py`）

这是代码库中**唯一消费** `desirePrediction` 的模块：

```python
desire_prediction = modelV2.meta.desirePrediction  # 展平的32维列表
# 取第一个时间步（index 3 和 4 位于 0-7 范围内）
l_lane_change_prob = desire_prediction[log.Desire.laneChangeLeft]   # index=3
r_lane_change_prob = desire_prediction[log.Desire.laneChangeRight]  # index=4

l_lane_close = left_lane_visible and (lane_lines[1].y[0] > -(1.08 + CAMERA_OFFSET))
r_lane_close = right_lane_visible and (lane_lines[2].y[0] < (1.08 - CAMERA_OFFSET))

self.left  = bool(l_lane_change_prob > LANE_DEPARTURE_THRESHOLD and l_lane_close)
self.right = bool(r_lane_change_prob > LANE_DEPARTURE_THRESHOLD and r_lane_close)
```

LDW 触发需同时满足：

| 条件 | 说明 |
|------|------|
| 车速 > 31mph | 低速不报警 |
| `recent_blinker = False` | 近5秒内**未拨转向灯**（有意变道则抑制） |
| `latActive = False` | 横向控制未激活（驾驶员手动驾驶） |
| 变道概率 > 0.1 | `desire_pred` 预测到明显变道趋势 |
| 对应侧车道线可见且距离很近 | 几何验证 |

核心逻辑：**视觉预测到变道，但驾驶员没有拨转向灯** → 判定为无意识偏道 → 触发报警。

#### 与策略网络 `desire_state` 的对比

两者命名相似但来源和语义完全不同：

| | `desire_pred`（视觉网络） | `desire_state`（策略网络） |
|---|---|---|
| **来源网络** | Vision Network | Policy Network |
| **形状** | `(4, 8)` 多时间步概率 | `(8,)` 单时间步概率 |
| **输入依据** | 纯摄像头图像 | 视觉特征 + desire_pulse（驾驶员操作） |
| **语义** | "车辆轨迹暗示将要发生什么变道？" | "驾驶员指令当前处于哪种意图状态？" |
| **用途** | LDW 无意识偏道检测 | DesireHelper 变道完成判断 |
| **下游** | `ldw.py` | `modeld.py:488`（反馈状态机） |

两者构成互补的意图感知体系：
- `desire_pred` 感知**无意识行为**（车辆正在漂移）
- `desire_state` 跟踪**有意识指令**（驾驶员主动变道中）

---

## 四、策略网络输入

策略网络**仅有三个输入张量**（由 `driving_policy_metadata.pkl` 精确定义），全部为过去/当前数据，不含任何未来预测：

```
desire_pulse:       (1, 25, 8)    过去5秒的意图脉冲历史
traffic_convention: (1, 2)        当前时刻的左/右舵标志
features_buffer:    (1, 25, 512)  过去5秒的视觉特征历史
```

策略网络输出的未来规划（`plan`，未来10秒）完全由**网络权重中学到的模式**生成，不依赖外部未来预测输入。

### 4.1 features_buffer（时序特征缓冲）

| 属性 | 值 |
|------|-----|
| 形状 | `(1, 25, 512)` |
| 数据类型 | float32 |
| 来源 | 视觉网络 `hidden_state` [1064:1576] |
| 语义 | 过去5秒的驾驶特征序列（5Hz上下文频率，25帧） |

**滑动窗口机制（InputQueues）**：

```
视觉网络：20Hz 运行
策略上下文：5Hz（每4帧取1帧）
窗口长度：25帧 × (1/5Hz) = 5秒历史

更新逻辑：
  每隔 MODEL_FREQ/MODEL_CONTEXT_FREQ = 4帧
  将最新 hidden_state 推入长度25的环形缓冲
```

### 4.2 desire_pulse（驾驶意图脉冲）

| 属性 | 值 |
|------|-----|
| 形状 | `(1, 25, 8)` |
| 数据类型 | float32 |
| 来源 | `DesireHelper` 状态机 → one-hot 编码 → 上升沿脉冲检测 → `InputQueues` 缓冲 |
| 语义 | 过去5秒内，驾驶意图发生**跳变**的时刻（one-hot 脉冲），其余时刻全零 |

#### 4.2.1 Desire 枚举语义（8维 one-hot 的索引）

| 索引 | 枚举名 | 含义 | 当前是否实际产生 |
|------|--------|------|----------------|
| 0 | `none` | 无特定意图（默认） | 是（绝大部分时间） |
| 1 | `turnLeft` | 左转 | 否（保留） |
| 2 | `turnRight` | 右转 | 否（保留） |
| 3 | `laneChangeLeft` | 向左变道中 | 是 |
| 4 | `laneChangeRight` | 向右变道中 | 是 |
| 5 | `keepLeft` | 保持左侧车道 | 否（保留） |
| 6 | `keepRight` | 保持右侧车道 | 否（保留） |
| 7 | (保留) | 未定义 | 否 |

当前状态机实际只会产生 `none`、`laneChangeLeft`、`laneChangeRight` 三种值。

#### 4.2.2 DesireHelper 状态机（`selfdrive/controls/lib/desire_helper.py`）

状态机驱动 desire 的变化，是 desire_pulse 的根本来源：

```
LaneChangeState.off  （初始/默认，desire = none）
        │
        │ 触发条件：
        │   ① 单侧转向灯刚亮（上升沿：prev=False → now=True）
        │   ② 车速 ≥ 20mph（8.9 m/s）
        │   ③ 横向控制（lateral_active）已激活
        ▼
LaneChangeState.preLaneChange  （等待确认，desire 仍 = none）
        │
        │ 触发条件：
        │   ① 方向盘扭矩施加（steeringPressed + 扭矩方向与变道方向一致）
        │   ② 无盲点检测（blindspot_detected = False）
        ▼
LaneChangeState.laneChangeStarting  ← desire = laneChangeLeft/Right ★
        │   lane_change_ll_prob：1.0 → 0.0（以 2×DT_MDL/帧 的速率，约0.5秒衰减完）
        │
        │ 触发条件（同时满足）：
        │   ① 模型预测 lane_change_prob < 0.02（策略网络认为变道已完成）
        │   ② lane_change_ll_prob < 0.01（车道线置信度已降至近零）
        ▼
LaneChangeState.laneChangeFinishing  ← desire 继续 = laneChangeLeft/Right
        │   lane_change_ll_prob：0.0 → 1.0（以 DT_MDL/帧 的速率，约1秒恢复）
        │
        │ 触发条件：lane_change_ll_prob > 0.99
        ▼
LaneChangeState.off（若转向灯已关）
或 preLaneChange（若转向灯仍亮，准备下次变道）
```

**DESIRES 映射表**（状态+方向 → desire 枚举值）：

```python
DESIRES = {
  LaneChangeDirection.none: {
    off: none,  preLaneChange: none,
    laneChangeStarting: none,  laneChangeFinishing: none,
  },
  LaneChangeDirection.left: {
    off: none,  preLaneChange: none,           # ← 准备阶段 desire 仍是 none
    laneChangeStarting:  laneChangeLeft,        # ← 实际变道时才非零
    laneChangeFinishing: laneChangeLeft,
  },
  LaneChangeDirection.right: { ... 对称 ... }
}
```

**关键设计**：`preLaneChange` 阶段（等待驾驶员确认转向）desire 保持 `none`，只有驾驶员真正施加转向扭矩后 desire 才跳变到 3/4，此时才产生脉冲。

**keep_pulse 机制**（`desire_helper.py:111`）：在 `preLaneChange` 状态下，`keep_pulse_timer` 每秒复位一次，用于将来 `keepLeft`/`keepRight` 意图的周期性脉冲管理（当前这两个 desire 值未被实际输出，该机制为预留设计）。

#### 4.2.3 One-Hot 编码（`modeld.py:442`）

```python
desire = DH.desire  # DesireHelper 输出的枚举整数，如 3 = laneChangeLeft

vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)  # shape: (8,)
if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
    vec_desire[desire] = 1.0

# 示例：desire=3 → vec_desire = [0, 0, 0, 1, 0, 0, 0, 0]
```

#### 4.2.4 上升沿脉冲检测（`modeld.py:265`，核心）

```python
# 每帧（20Hz）执行一次
inputs['desire_pulse'][0] = 0   # 归零 batch 维
new_desire = np.where(
    inputs['desire_pulse'] - self.prev_desire > 0.99,  # 从 0 跳变到 1
    inputs['desire_pulse'],                             # 跳变时输出 1
    0                                                   # 未跳变时输出 0
)
self.prev_desire[:] = inputs['desire_pulse']           # 记录当前帧供下帧比较
```

**时序示意（以左变道为例）**：

```
帧   desire(DH)    vec_desire       prev_desire      new_desire(脉冲)
 1   none          [0,0,0,0,…]      [0,0,0,0,…]      [0,0,0,0,…]
 2   none          [0,0,0,0,…]      [0,0,0,0,…]      [0,0,0,0,…]
     ─── 驾驶员拨左转向灯 + 施加转向扭矩 → 进入 laneChangeStarting ───
 3   laneChangeL   [0,0,0,1,…]      [0,0,0,0,…]      [0,0,0,1,…] ★ 脉冲
 4   laneChangeL   [0,0,0,1,…]      [0,0,0,1,…]      [0,0,0,0,…]   无脉冲
 5   laneChangeL   [0,0,0,1,…]      [0,0,0,1,…]      [0,0,0,0,…]   无脉冲
     ─── 变道完成，回到 off ───
 6   none          [0,0,0,0,…]      [0,0,0,1,…]      [0,0,0,0,…]   无脉冲（下降沿不触发）
 7   none          [0,0,0,0,…]      [0,0,0,0,…]      [0,0,0,0,…]
```

**设计意图**：策略网络具有类似 RNN 的内部记忆，若将持续的 desire 直接喂入，网络状态会被持续驱动并可能累积偏移。用脉冲信号只在意图**开始时触发一次**，让网络自行维持内部状态，是更鲁棒的设计。

#### 4.2.5 InputQueues 缓冲与 pulse 聚合（`modeld.py:123`）

`InputQueues` 负责处理视觉网络（20Hz）与策略上下文频率（5Hz）之间 4:1 的速率差异：

```python
# 内部缓冲区形状：(1, 100, 8)  = (batch, 25策略帧×4视觉帧, desire维)
# enqueue：将最新脉冲帧写入缓冲末尾（左移淘汰最旧）

# get() 时的 pulse 专用聚合（modeld.py:190）：
out[k] = self.q[k].reshape(1, 25, 4, 8).max(axis=2)
#                          ↑    ↑   ↑
#                        batch 策略帧 每策略帧对应的4个视觉帧
# → shape: (1, 25, 8)
# 语义：每个策略时间步（0.2秒）内，4帧中只要有任意一帧产生了脉冲，输出就为 1
```

**聚合意义**：采用 `max` 而非采样，确保脉冲不因降采样而丢失。即使脉冲恰好落在被跳过的视觉帧上，策略网络也能感知到。

#### 4.2.6 闭环反馈：策略网络输出反哺状态机

策略网络输出的 `desire_state`（8维 softmax）**反馈给** DesireHelper，关闭控制环：

```python
# modeld.py:484
desire_state = modelv2_send.modelV2.meta.desireState
lane_change_prob = desire_state[log.Desire.laneChangeLeft] \
                 + desire_state[log.Desire.laneChangeRight]
DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
```

在 `laneChangeStarting` 状态内，`lane_change_prob < 0.02` 是进入 `laneChangeFinishing` 的必要条件之一（`desire_helper.py:87`）。即**模型自己预测"变道概率低于2%"时，才认为变道完成**，形成如下闭环：

```
驾驶员转向 → DesireHelper → desire_pulse → 策略网络（plan）
                 ↑                               ↓
          lane_change_prob ←── desire_state（softmax 8维）
```

#### 4.2.7 各层数据汇总

| 层次 | 变量 | 形状 | 语义 |
|------|------|------|------|
| DesireHelper 输出 | `DH.desire` | 标量 int | 当前意图枚举值（0-7） |
| One-hot 编码 | `vec_desire` | `(8,)` | 稀疏向量，当前意图位为 1.0 |
| 上升沿检测后 | `new_desire` | `(8,)` | **仅在意图跳变时为 1**，其余全零 |
| 队列内部缓冲 | `q['desire_pulse']` | `(1, 100, 8)` | 最近 100 个视觉帧（5秒）的脉冲记录 |
| 策略网络输入 | `desire_pulse` | `(1, 25, 8)` | 25个策略时间步，每步取4帧内 max |

### 4.3 traffic_convention（交通习惯）

| 属性 | 值 |
|------|-----|
| 形状 | `(1, 2)` |
| 数据类型 | float32 |
| 来源 | `driverMonitoringState.isRHD` |
| 语义 | 左舵/右舵驾驶规则 |

```python
# 左舵（LHD，中国/美国等）
traffic_convention = [1.0, 0.0]

# 右舵（RHD，英国/日本/澳大利亚等）
traffic_convention = [0.0, 1.0]
```

### 4.4 输入的时序性与"未来规划"的来源

策略网络的三个输入均为**历史或当前数据**，没有未来预测作为输入。网络输出的未来10秒 `plan` 完全来自权重中学到的驾驶模式。

#### 输入时间覆盖范围

```
         ←─────────── 过去 5 秒 ───────────→  当前
时间轴:  [t-5s][t-4.8s]...[t-0.4s][t-0.2s][t]
帧编号:   [0]   [1]          [23]    [24]
                              ↑
              features_buffer 和 desire_pulse 均覆盖这25帧
```

#### `desire_pulse` 的时间位置编码

25帧历史窗口让网络能感知脉冲的**时间位置**，从而推断变道处于哪个阶段：

```
示例A：脉冲在 24 帧前（5秒前），变道早已完成，plan 已回归直行
示例B：脉冲在 2 帧前（0.4秒前），变道刚开始，plan 输出横向偏移轨迹
示例C：脉冲在 12 帧前（2.4秒前），变道进行到一半，plan 处于横向过渡段
```

脉冲的**位置**隐式编码了"指令发出了多久"，网络据此估计当前应处于变道的哪个阶段，并输出对应的未来轨迹。

#### `desire_state` 是输出而非输入

容易混淆的是：`desire_state` 是策略网络的**输出**（8维 softmax），而非输入。网络从历史数据中推断当前意图状态，再通过反馈环传给 DesireHelper。整个系统没有"未来意图"作为先验输入。

| 字段 | 方向 | 含义 |
|------|------|------|
| `desire_pulse` | → 输入 | 过去5秒内驾驶员何时发出了哪种指令 |
| `desire_state` | ← 输出 | 当前时刻网络估计的意图状态概率 |
| `desire_pred` | ← 视觉网络输出 | 视觉预测的未来变道趋势（仅用于 LDW） |

---

## 五、策略网络输出（1000 维）

策略网络原始输出为 1000 维向量，按以下语义切片解析：

```
[0   : 990]  plan          990维  未来行驶规划（MHP+MDN）
[990 : 998]  desire_state    8维  当前意图分布（Softmax）
```

### 5.1 plan（行驶规划，990维）

**MHP（5假设）+ MDN 格式解析**：

```python
# plan = 5假设 × (33时间点 × 15维状态均值 + 33×15维标准差) + 5维权重
# 选择权重最高的假设作为主输出
# 最终形状：(33, 15)
```

**33个时间点（T_IDXS，非线性分布）**：

```python
# 覆盖未来 0 ~ 10 秒
T_IDXS[i] = 10.0 * (i / 32)²
# 近端密集，远端稀疏：约 [0, 0.01, 0.04, ..., 9.77, 10.0]
```

**15维状态向量（Plan enum）**：

| 字段 | 索引 | 维度 | 单位 | 含义 |
|------|------|------|------|------|
| `POSITION` | 0:3 | 3 | 米 | 位置 [x前, y左, z上] |
| `VELOCITY` | 3:6 | 3 | m/s | 速度分量 |
| `ACCELERATION` | 6:9 | 3 | m/s² | 加速度分量 |
| `T_FROM_CURRENT_EULER` | 9:12 | 3 | rad | 由当前起的欧拉角增量 [Δroll, Δpitch, Δyaw] |
| `ORIENTATION_RATE` | 12:15 | 3 | rad/s | 姿态变化率 [roll_rate, pitch_rate, yaw_rate] |

坐标系：**标定坐标系**（Calib frame，x前、y左、z上）

### 5.2 desire_state（当前意图分布）

```
形状：(1, 8)  float32
激活：Softmax（概率和为1）
含义：当前时刻对8种驾驶意图的概率估计
用途：更新 laneChangeState / laneChangeDirection
```

---

## 六、动作生成（Plan → Action）

代码：`selfdrive/modeld/modeld.py`，`selfdrive/controls/lib/drive_helpers.py`

### 6.1 期望加速度（desiredAcceleration）

从 plan 速度序列通过运动学反推：

```python
def get_accel_from_plan(plan, v_ego, CP):
    # 纵向延迟补偿
    action_t = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS + DT_MDL

    v_now    = plan[0, VELOCITY, 0]       # 当前时刻速度
    a_now    = plan[0, ACCELERATION, 0]   # 当前时刻加速度
    v_target = interp(action_t, T_IDXS, plan[:, VELOCITY, 0])  # 插值目标速度

    # 运动学反推：考虑延迟后的加速度
    a_target = 2 * (v_target - v_now) / action_t - a_now

# 一阶低通滤波（纵向平滑）
alpha = 1 - exp(-DT_MDL / LONG_SMOOTH_SECONDS)  # LONG_SMOOTH_SECONDS = 0.3s
a_smoothed = alpha * a_target + (1 - alpha) * prev_a
```

### 6.2 期望曲率（desiredCurvature）

从 plan 航向角序列反推：

```python
def get_curvature_from_plan(plan, v_ego, sm):
    # 横向延迟补偿
    lat_delay  = sm["liveDelay"].lateralDelay
    action_t   = lat_delay + LAT_SMOOTH_SECONDS + DT_MDL  # LAT_SMOOTH_SECONDS = 0.0

    psi_target = interp(action_t, T_IDXS,
                        plan[:, T_FROM_CURRENT_EULER, 2])  # yaw增量
    psi_rate   = plan[0, ORIENTATION_RATE, 2]              # yaw角速度

    # 曲率 = 航向变化 / 行驶距离 ≈ yaw_rate / v
    curvature  = 2 * psi_target / (v_ego * action_t) - psi_rate / v_ego

# 低速保护：速度 < 0.3 m/s 时保持前一时刻曲率
if v_ego <= MIN_LAT_CONTROL_SPEED:
    curvature = prev_curvature
```

---

## 七、消息发布与消费

### 7.1 模型进程发布的消息

| 消息名 | 频率 | 主要内容 | 下游消费方 |
|--------|------|---------|-----------|
| `modelV2` | 20 Hz | 完整模型输出（规划+感知+元数据） | controlsd, plannerd, selfdrived, UI |
| `drivingModelData` | 20 Hz | 精简版（前车+车道线+规划） | controlsd |
| `cameraOdometry` | 20 Hz | 相机运动估计 | locationd |

### 7.2 modelV2 消息结构

```
modelV2:
  frameId / frameIdExtra / frameAge
  modelExecutionTime (float)

  # 行驶规划（来自 policy plan）
  position:       XYZTData[33]  # x,y,z + 时间戳 + 标准差
  velocity:       XYZTData[33]
  acceleration:   XYZTData[33]
  orientation:    XYZTData[33]  # 欧拉角
  orientationRate: XYZTData[33]

  # 环境感知（来自 vision）
  laneLines:      [XYZTData; 4]   # 4条车道线各33点
  laneLineProbs:  [float; 4]      # 车道线存在概率
  roadEdges:      [XYZTData; 2]   # 2条路沿
  roadEdgeStds:   [float; 33]

  # 前车（来自 vision lead）
  leadsV3:        [LeadDataV3; 3] # 最多3个目标
    .prob:     float              # 存在概率
    .x/y/v/a:  [float; 6]        # 6个时间点状态
    .t:        [0,2,4,6,8,10]    # 秒

  # 元数据（来自 vision meta + policy desire）
  meta:
    desireState:          [float; 8]   # policy desire_state
    desirePrediction:     [float; 32]  # vision desire_pred（4×8）
    engagedProb:          float
    disengagePredictions:
      brakeDisengageProbs:  [float; 5]  # 2/4/6/8/10秒
      gasDisengageProbs:    [float; 5]
      steerOverrideProbs:   [float; 5]
      brake3/4/5MetersPerSecondSquaredProbs: [float; 5]
      gasPressProbs:        [float; 6]  # 0/2/4/6/8/10秒
      brakePressProbs:      [float; 6]
    hardBrakePredicted:   bool        # FCW前碰预警
    laneChangeState:      enum
    laneChangeDirection:  enum
    confidence:           {red/yellow/green}

  # 控制动作
  action:
    desiredCurvature:     float  # 1/m
    desiredAcceleration:  float  # m/s²
    shouldStop:           bool
```

### 7.3 cameraOdometry 消息结构

```
cameraOdometry:
  trans:     [float; 3]  # [tx, ty, tz] 平移速度 m/s（设备坐标系）
  rot:       [float; 3]  # [rx, ry, rz] 旋转角速度 rad/s
  transStd:  [float; 3]  # 标准差
  rotStd:    [float; 3]

  wideFromDeviceEuler:    [float; 3]  # 广角相机对齐欧拉角
  roadTransformTrans:     [float; 3]  # 路面坐标变换平移
  roadTransformTransStd:  [float; 3]
```

---

## 八、上游依赖

模型进程订阅以下外部消息：

| 消息来源 | 消息名 | 用途 |
|----------|--------|------|
| `camerad` | VisionIPC 帧 | 主/广角相机原始图像（零拷贝） |
| `calibrationd` | `liveCalibration` | rpyCalib → 透视变换矩阵 |
| `controlsd` | `liveDelay` | 横向延迟补偿（lateralDelay） |
| `drivermonitoringd` | `driverMonitoringState` | isRHD → traffic_convention |
| `selfdrived` | `carState` | vEgo → 低速曲率保护 |
| `carparams` | `CarParams` | longitudinalActuatorDelay → 纵向延迟 |

---

## 九、前碰撞预警（FCW）逻辑

```python
# fill_model_msg.py

# 维护滚动缓冲
prev_brake_5ms2_probs: 最近5帧的 brake5MetersPerSecondSquaredProbs[0]
prev_brake_3ms2_probs: 最近2帧的 brake3MetersPerSecondSquaredProbs[0]

# 触发条件（需同时满足）
hard_brake_predicted = (
    all(prev_brake_5ms2_probs > [0.05, 0.05, 0.15, 0.15, 0.15])
    and
    all(prev_brake_3ms2_probs > [0.7, 0.7])
)
```

---

## 十、置信度评估

每2秒采样一次，基于脱离预测概率计算：

```python
# 综合脱离概率（任一事件发生）
any_disengage = 1 - (1 - brake_disengage) * (1 - gas_disengage) * (1 - steer_override)

# 条件独立脱离概率
ind_disengage = diff(any_disengage) / (1 - any_disengage[:-1])

# 滚动加权评分（5帧缓冲）
score = weighted_sum(disengage_buffer)

# 三级置信度阈值
green  < 0.01165  # 正常，显示绿色
yellow < 0.06157  # 注意，显示黄色
else   → red      # 警告，显示红色
```

---

## 十一、下游控制集成

### 11.1 controlsd（横向控制）

```python
# 订阅 modelV2
new_desired_curvature = sm['modelV2'].action.desiredCurvature

# 限幅处理（考虑车速和侧倾）
desired_curvature, limited = clip_curvature(
    v_ego, prev_curvature, new_desired_curvature, roll
)

# 发送给横向控制器（INDI / LQR / PID）
actuators.steer = lat_controller.update(desired_curvature)

# 车道变道控制
meta = sm['modelV2'].meta
if meta.laneChangeState != LaneChangeState.off:
    CC.leftBlinker  = (meta.laneChangeDirection == left)
    CC.rightBlinker = (meta.laneChangeDirection == right)
```

### 11.2 plannerd（纵向规划）

```python
# 订阅 modelV2
model = sm['modelV2']

# 模型规划直接使用
position     = model.position.x      # 33点 x坐标序列
velocity     = model.velocity.x      # 33点速度序列
acceleration = model.acceleration.x  # 33点加速度序列

# 前车数据
leads = model.leadsV3  # 最多3个目标

# 送入MPC优化器
LongitudinalMpc.update(position, velocity, acceleration, leads)

# 输出 longitudinalPlan → controlsd
```

### 11.3 locationd（传感器融合）

```python
# 订阅 cameraOdometry
cam_odo = sm['cameraOdometry']

# 相机运动作为里程计约束输入卡尔曼滤波
locationd.ekf.observe_camera_odometry(
    trans=cam_odo.trans,
    rot=cam_odo.rot,
    transStd=cam_odo.transStd,
    rotStd=cam_odo.rotStd
)
```

---

## 十二、关键常量与配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `MODEL_RUN_FREQ` | 20 Hz | 视觉和策略网络运行频率 |
| `MODEL_CONTEXT_FREQ` | 5 Hz | 策略上下文采样频率 |
| `N_FRAMES` | 2 | 视觉输入时序帧数 |
| `FEATURE_LEN` | 512 | 隐藏状态维度 |
| `DESIRE_LEN` | 8 | 意图类型数量 |
| `IDX_N` | 33 | 时间索引点数 |
| `PLAN_MHP_N` | 5 | plan 假设数量 |
| `PLAN_MHP_SELECTION` | 1 | 选择权重最高的1个假设 |
| `LEAD_MHP_N` | 2 | lead 假设数量 |
| `LEAD_MHP_SELECTION` | 3 | 选择权重最高的3个目标 |
| `DT_MDL` | 0.05 s | 模型帧间隔 |
| `LONG_SMOOTH_SECONDS` | 0.3 s | 纵向平滑时间常数 |
| `LAT_SMOOTH_SECONDS` | 0.0 s | 横向平滑时间常数 |
| `MIN_LAT_CONTROL_SPEED` | 0.3 m/s | 低速曲率保护阈值 |

---

## 十三、坐标系说明

| 坐标系 | X轴 | Y轴 | Z轴 | 用途 |
|--------|-----|-----|-----|------|
| **Device** | 前 | 右 | 下 | 物理传感器安装参考 |
| **View** | 右 | 下 | 前 | OpenCL 图像处理中间态 |
| **Calib** | 前 | 左 | 上 | 消除安装偏差后的规范坐标 |
| **Model** | — | — | — | 虚拟相机像素空间 |

**模型输出坐标系**：plan 在 **Calib（标定坐标系）** 下，pose 在 **Device 坐标系** 下。

---

## 十四、关键代码索引

| 文件 | 功能 |
|------|------|
| `selfdrive/modeld/modeld.py` | 主循环、模型加载、推理、消息发布 |
| `selfdrive/modeld/parse_model_outputs.py` | MDN/Softmax/Sigmoid 解码 |
| `selfdrive/modeld/fill_model_msg.py` | cereal 消息填充、FCW、置信度 |
| `selfdrive/modeld/constants.py` | T_IDXS、Plan enum、输出切片常量 |
| `selfdrive/modeld/models/commonmodel.cc` | C++ 帧处理、OpenCL 缓冲管理 |
| `selfdrive/modeld/transforms/transform.cl` | OpenCL 透视变换内核 |
| `selfdrive/modeld/transforms/loadyuv.cl` | OpenCL YUV重排内核 |
| `selfdrive/modeld/runners/tinygrad_helpers.py` | QCOM 显存零拷贝映射 |
| `selfdrive/controls/lib/drive_helpers.py` | plan → 曲率/加速度转换 |
| `selfdrive/controls/controlsd.py` | 横向控制（消费 desiredCurvature） |
| `selfdrive/controls/plannerd.py` | 纵向规划（消费 plan） |
| `selfdrive/locationd/locationd.py` | 传感器融合（消费 cameraOdometry） |

---

## 十五、数据流总结

```
摄像头 (20Hz)
  ↓ YUV420帧
  ↓ [OpenCL 透视变换 + YUV重排]
  ↓ (1,12,128,256) uint8

视觉网络 → 1576维
  ├─ 感知：车道线(528) + 路沿(264) + 前车(144+3) + 位姿(12+6+12)
  ├─ 元数据：meta(55) + desire_pred(32)
  └─ 特征：hidden_state(512) ──┐
                                ↓ 入队
策略网络输入:                  features_buffer (25帧×512)
  ├─ features_buffer (1,25,512)
  ├─ desire_pulse    (1,25,8)  ← DesireHelper (转向灯/方向盘)
  └─ traffic_convention (1,2) ← driverMonitoringState

策略网络 → 1000维
  ├─ plan (990) → 未来10秒规划（5假设，选最优）
  └─ desire_state (8) → 当前意图分布

动作生成:
  ├─ desiredCurvature  = f(plan.yaw, v_ego, lat_delay)
  └─ desiredAcceleration = f(plan.velocity, v_ego, long_delay)

消息发布:
  ├─ modelV2       → controlsd + plannerd + selfdrived + UI
  ├─ drivingModelData → controlsd
  └─ cameraOdometry → locationd
```
