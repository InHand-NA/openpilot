# 卡车 LDW/FCW 视觉模型训练技术方案

## 目录

- [第1章：项目背景与需求分析](#第1章项目背景与需求分析)
- [第2章：openpilot LDW/FCW 实现分析](#第2章openpilot-ldwfcw-实现分析)
- [第3章：模型输出精简设计](#第3章模型输出精简设计)
- [第4章：网络架构设计](#第4章网络架构设计)
- [第5章：训练策略](#第5章训练策略)
- [第6章：损失函数设计](#第6章损失函数设计)
- [第7章：相机标定与高度估计](#第7章相机标定与高度估计)
- [第8章：模型部署方案](#第8章模型部署方案)
- [第9章：LDW/FCW 告警逻辑](#第9章ldwfcw-告警逻辑)
- [第10章：项目里程碑](#第10章项目里程碑)

---

## 第1章：项目背景与需求分析

### 1.1 openpilot supercombo 模型架构概述

openpilot 的驾驶模型采用**双网络架构**（`selfdrive/modeld/modeld.py`），ONNX 文件位于 `selfdrive/modeld/models/`：

1. **视觉网络（`driving_vision.onnx`）**：23M 参数，45MB（float16）。接收双目相机图像（窄角 `img` + 广角 `big_img`），backbone 为 **ConvNeXt 变体**（双尺度串行深度卷积），输出车道线、前车、视觉里程计等感知结果，以及一个 512 维的 `hidden_state` 特征向量。输出 **1576D**。
2. **策略网络（`driving_policy.onnx`）**：6.9M 参数，14MB（float16）。接收视觉网络的 `hidden_state`（25 帧缓冲）、驾驶意图脉冲（`desire_pulse`）和交通规则信号，使用 **1 层 GPT 风格 Transformer** 融合时序信息，输出行驶轨迹规划（`plan`）和意图状态（`desire_state`）。输出 **1000D**。

两个网络分别以独立的 `.pkl` 文件存储（`selfdrive/modeld/modeld.py:67-68`）：
```
VISION_PKL_PATH  → models/driving_vision_tinygrad{_suffix}.pkl
POLICY_PKL_PATH  → models/driving_policy_tinygrad{_suffix}.pkl
```

**关键参数**（`selfdrive/modeld/constants.py:6-26`）：
- 输入帧数：`N_FRAMES = 2`（双帧时序输入）
- 模型运行频率：`MODEL_RUN_FREQ = 20` Hz
- 模型上下文频率：`MODEL_CONTEXT_FREQ = 5` Hz
- 特征维度：`FEATURE_LEN = 512`（hidden_state 维度）
- 意图编码：`DESIRE_LEN = 8`
- 时空采样点：`IDX_N = 33`（覆盖前方 0-192m，`X_IDXS`），时间覆盖 0-10s（`T_IDXS`）

### 1.2 LDW/FCW 对模型输出的最小依赖

**LDW（车道偏离预警）** 仅依赖视觉网络的以下输出：
- 车道线几何（`lane_lines`）：4 条线 × 33 个采样点 × 2 维（y, z）
- 车道线概率（`lane_lines_prob`）：4 个概率值
- 意图预测（`desire_pred`）：4 个时间片 × 8 个意图类别

**FCW（前方碰撞预警）** 依赖：
- 前车轨迹预测（`lead`）：检测到的前车的距离/速度/加速度
- 前车存在概率（`lead_prob`）
- 急刹预测概率（`meta` 中的 `HARD_BRAKE_3`/`HARD_BRAKE_5`）

**关键发现**：LDW 和 FCW 所需的全部输出均来自**视觉网络**，不依赖策略网络。因此卡车产品可以**完全移除策略网络**，大幅降低模型体积和计算量。

### 1.3 卡车场景特殊性

| 维度 | 乘用车 (openpilot) | 卡车 |
|------|---------------------|------|
| 相机配置 | 双目：fcam（窄角，focal 2648）+ ecam（广角，focal 567） | **单目 fcam（窄角）** |
| 相机安装高度 | ~1.22m | 2.0-2.8m |
| 车身宽度 | ~1.8m | ~2.5m |
| 制动距离 (80km/h) | ~36m | ~70m+ |
| 载重变化 | 小 | 空载/满载高度差 10-20cm |
| FOV 特点 | 标准视角 | 高视角，更多地面可见 |
| 功能需求 | ACC + ALC（控制） | LDW + FCW（仅告警） |
| 部署硬件 | comma 3X (Qualcomm) | CPU/NPU 嵌入式平台 |

**相机选型说明**：卡车方案采用**单目窄角相机（fcam）**，原因如下：
- openpilot 的双目设计并非立体视觉（不靠视差测距），而是窄角（~40° FOV）+ 广角（~120° FOV）互补覆盖
- LDW 和 FCW 的核心感知需求（车道线检测、前车测距）主要依赖窄角相机的远距离分辨能力
- 窄角 fcam 的 focal length 为 2648，远距离目标（60-100m+）的像素分辨率远优于广角 ecam
- 单目方案简化硬件和系统复杂度，降低成本，适合第一阶段快速验证
- 后续如需扩展近距离感知能力（如弯道大角度场景），可增加广角相机升级为双目

---

## 第2章：openpilot LDW/FCW 实现分析

### 2.1 LDW 实现详解

**源码位置**：`selfdrive/controls/lib/ldw.py`

```python
CAMERA_OFFSET = 0.04          # 相机偏离车辆中心线的横向偏移（米）
LDW_MIN_SPEED = 31 * CV.MPH_TO_MS  # ≈ 13.86 m/s ≈ 49.9 km/h
LANE_DEPARTURE_THRESHOLD = 0.1     # 意图概率阈值
```

**判断逻辑**（`ldw.py:16-37`）：

1. **速度门限**：`vEgo > LDW_MIN_SPEED`（约 50 km/h）
2. **转向灯冷却**：转向灯信号后 5 秒内不触发
3. **非主动控制**：`not CC.latActive`
4. **车道线可见性**：`laneLineProbs[1] > 0.5`（左线）或 `laneLineProbs[2] > 0.5`（右线）
5. **意图概率**：`desirePrediction[laneChangeLeft/Right] > 0.1`
6. **距离阈值**：
   - 左偏离：`lane_lines[1].y[0] > -(1.08 + CAMERA_OFFSET)` 即 `> -1.12m`
   - 右偏离：`lane_lines[2].y[0] < (1.08 - CAMERA_OFFSET)` 即 `< 1.04m`

其中 `lane_lines[N].y[0]` 是第 N 条车道线在 `x=0`（车辆当前位置）处的**横向偏移**（米），正值为右。`lane_lines[1]` 和 `lane_lines[2]` 分别对应 ego 车道的左边界和右边界。

### 2.2 FCW 双机制

FCW 在 openpilot 中通过两个独立机制触发，任一满足即告警：

#### 机制1：模型预测急刹（端到端）

**源码位置**：`selfdrive/modeld/fill_model_msg.py:143-149`

模型 `meta` 输出中包含未来 2/4/6/8/10 秒的急刹概率。FCW 使用两个滑动窗口判定：

```python
# 5帧滑动窗口（-5 m/s² 急刹概率）
FCW_THRESHOLDS_5MS2 = [0.05, 0.05, 0.15, 0.15, 0.15]  # constants.py:29
# 2帧滑动窗口（-3 m/s² 急刹概率）
FCW_THRESHOLDS_3MS2 = [0.7, 0.7]                        # constants.py:30

# fill_model_msg.py:143-149
publish_state.prev_brake_5ms2_probs[:-1] = publish_state.prev_brake_5ms2_probs[1:]
publish_state.prev_brake_5ms2_probs[-1] = net_output_data['meta'][0, Meta.HARD_BRAKE_5][0]
publish_state.prev_brake_3ms2_probs[:-1] = publish_state.prev_brake_3ms2_probs[1:]
publish_state.prev_brake_3ms2_probs[-1] = net_output_data['meta'][0, Meta.HARD_BRAKE_3][0]
hard_brake_predicted = (publish_state.prev_brake_5ms2_probs > FCW_THRESHOLDS_5MS2).all() and \
    (publish_state.prev_brake_3ms2_probs > FCW_THRESHOLDS_3MS2).all()
```

**触发条件**：连续 5 帧的 HARD_BRAKE_5（-5m/s²）概率依次超过 `[0.05, 0.05, 0.15, 0.15, 0.15]`，**且**连续 2 帧的 HARD_BRAKE_3（-3m/s²）概率超过 `[0.7, 0.7]`。

`Meta` 输出的切片定义（`constants.py:74-87`）：
```python
class Meta:
  ENGAGED = slice(0, 1)
  GAS_DISENGAGE = slice(1, 31, 6)   # 每6个元素取1个 → 5个时间点
  BRAKE_DISENGAGE = slice(2, 31, 6)
  STEER_OVERRIDE = slice(3, 31, 6)
  HARD_BRAKE_3 = slice(4, 31, 6)    # -3 m/s² 急刹概率（5个时间点）
  HARD_BRAKE_4 = slice(5, 31, 6)
  HARD_BRAKE_5 = slice(6, 31, 6)    # -5 m/s² 急刹概率（5个时间点）
  GAS_PRESS = slice(31, 55, 4)
  BRAKE_PRESS = slice(32, 55, 4)
```

meta 总维度 = 1（ENGAGED）+ 30（6类×5时间点）+ 24（4类×6时间点）= **55D**。

#### 机制2：物理 TTC 碰撞检测

**源码位置**：
- MPC 碰撞计数：`selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py:396-400`
- 纵向规划器 FCW 判定：`selfdrive/controls/lib/longitudinal_planner.py:151`

```python
# long_mpc.py:396-400
if (np.any(lead_xv_0[FCW_IDXS, 0] - self.x_sol[FCW_IDXS, 0] < CRASH_DISTANCE) and
        radarstate.leadOne.modelProb > 0.9):
    self.crash_cnt += 1
else:
    self.crash_cnt = 0

# longitudinal_planner.py:151
self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
```

其中 `CRASH_DISTANCE = 0.25m`，`FCW_IDXS = T_IDXS < 5.0`（前 5 秒内的时间点），当 MPC 求解发现 ego 车辆在未来 5 秒内任一时刻与前车距离小于 0.25m，**且前车置信度 > 0.9**，则 `crash_cnt` 递增。连续超过 2 次触发 FCW。

### 2.3 纯视觉前车检测

**源码位置**：`selfdrive/controls/radard.py:141-156`

当无雷达匹配到视觉目标时，直接使用模型输出构建 lead 状态：

```python
def get_RadarState_from_vision(lead_msg, v_ego, model_v_ego):
    lead_v_rel_pred = lead_msg.v[0] - model_v_ego
    return {
        "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),  # RADAR_TO_CAMERA = 1.52m
        "yRel": float(-lead_msg.y[0]),
        "vRel": float(lead_v_rel_pred),
        "vLead": float(v_ego + lead_v_rel_pred),
        "vLeadK": float(v_ego + lead_v_rel_pred),
        "aLeadK": float(lead_msg.a[0]),
        "aLeadTau": 0.3,
        "fcw": False,
        "modelProb": float(lead_msg.prob),
        "status": True,
        "radar": False,
    }
```

**卡车产品说明**：卡车方案无雷达，完全依赖此纯视觉路径。模型输出 `lead` 为纯 MDN 编码（非 MHP），包含 3 个时间偏移（0/2/4s）× 6 个时间点（0/2/4/6/8/10s）× 4 维状态（x, y, v, a）的均值和标准差，共 144D。以及 3 个 lead 选择的存在概率（3D）。

---

## 第3章：模型输出精简设计

### 3.1 openpilot 原始输出结构

视觉网络和策略网络的输出通过 `parse_model_outputs.py` 解析：

**视觉网络输出**（`parse_model_outputs.py:95-121`）：

| 输出名 | 解析方式 | 形状 | 说明 |
|--------|----------|------|------|
| `pose` | MDN | (6,) mean + (6,) std | 相机位姿（trans xyz + rot rpy） |
| `wide_from_device_euler` | MDN | (3,) mean + (3,) std | 广角相机欧拉角 |
| `road_transform` | MDN | (6,) mean + (6,) std | 路面变换参数 |
| `lane_lines` | MDN | (4, 33, 2) mean + std | 4条车道线的 y/z 坐标 |
| `road_edges` | MDN | (2, 33, 2) mean + std | 2条路缘线的 y/z 坐标 |
| `lane_lines_prob` | BCE sigmoid | (8,) → 取奇数索引 (4,) | 车道线存在概率 |
| `desire_pred` | CE softmax | (4, 8) | 未来意图预测 |
| `meta` | BCE sigmoid | (55,) | 元事件概率集合 |
| `lead_prob` | BCE sigmoid | (3,) | 前车存在概率 |
| `lead` | MDN | (3, 6, 4) mean + (3, 6, 4) std = 144D | 前车轨迹预测 |
| `hidden_state` | 直接传递 | (512,) | 传递给策略网络 |

**策略网络输出**（`parse_model_outputs.py:123-128`）：

| 输出名 | 解析方式 | 形状 | 说明 |
|--------|----------|------|------|
| `plan` | MDN | (1, 33, 15) mean + (1, 33, 15) std = 990D | 驾驶轨迹规划 |
| `desire_state` | CE softmax | (8,) | 当前意图状态 |

### 3.2 LDW 必需输出

| 输出 | 原始维度 | MDN 后维度 | 来源 | 用途 |
|------|----------|------------|------|------|
| `lane_lines` | 4×33×2 = 264 | 528 (含 std) | 视觉网络 | ego 左右车道线在各距离点的 y/z 坐标 |
| `lane_lines_prob` | 8 (raw logits) | 8 | 视觉网络 | 每条线的检测概率（取 `[1::2]` 共 4 个） |
| `desire_pred` | 4×8 = 32 | 32 | 视觉网络 | 车道变换意图预测（softmax） |

### 3.3 FCW 必需输出

| 输出 | 原始维度 | MDN 后维度 | 来源 | 用途 |
|------|----------|------------|------|------|
| `lead` | 3×6×4 = 72 | 144 (mean + std) | 视觉网络 | 前车 (x,y,v,a) × 6 时间点 × 3 选择 |
| `lead_prob` | 3 | 3 | 视觉网络 | 前车存在概率 |
| `meta` (FCW 相关) | 55 total, 取 10 | 10 | 视觉网络 | HARD_BRAKE_3/5 各 5 个时间点 |

### 3.4 辅助输出

| 输出 | MDN 后维度 | 用途 |
|------|------------|------|
| `pose` | 12 (含 std) | 视觉里程计，支持在线标定和时序一致性 |
| `road_transform` | 12 (含 std) | 路面几何/高度在线估计 |
| `road_edges` | 264 (含 std) | 路缘检测，辅助 LDW 判断（无标线场景） |

### 3.5 可完全移除的输出

| 输出 | 维度 | 移除理由 |
|------|------|----------|
| `plan` | 1×33×15 MDN → 990D | 策略网络输出，控制规划用。仅告警不需要 |
| `desire_state` | 8D | 策略网络输出，车道变换状态机用 |
| `wide_from_device_euler` | 6D | 双目对齐用，卡车单目 fcam 方案不需要 |
| `hidden_state` | 512D | 策略网络输入。不需策略网络时无意义 |
| `meta` (非 FCW 部分) | ~45D | ENGAGED、GAS/BRAKE_DISENGAGE、STEER_OVERRIDE 等，仅控制用 |

**精简后总输出维度估算**：

| 输出 | 维度（含 mean + std） |
|------|----------------------|
| lane_lines | 528 |
| lane_lines_prob | 8 |
| desire_pred | 32 |
| lead (纯 MDN) | 3×6×4×2 = 144 |
| lead_prob | 3 |
| meta (FCW only) | 10 |
| pose | 12 |
| road_transform | 12 |
| road_edges | 264 |
| **总计** | **~1013D** |

对比原始双网络总输出（视觉 1576D + 策略 1000D = 2576D），**减少约 60%**。

### 3.6 策略网络整体移除

卡车产品不做自动驾驶控制（ACC + ALC），仅做 LDW/FCW 告警。因此：

- 策略网络**整体移除**，包括其全部输入（`features_buffer`、`desire_pulse`、`traffic_convention`）和输出（`plan`、`desire_state`）
- `hidden_state` 不再需要从视觉网络传递
- `InputQueues`（`modeld.py:123-196`）中为策略网络维护的时序缓冲也可移除

---

## 第4章：网络架构设计

### 4.1 openpilot 视觉网络实际架构（ONNX 分析）

基于 `selfdrive/modeld/models/driving_vision.onnx` 的逆向分析，openpilot 视觉网络的实际架构如下：

```
Input: img [1,12,128,256] (uint8) + big_img [1,12,128,256] (uint8)
       ↓ Cast(uint8→fp16)
       ↓ Concat(axis=1) → [1, 24, 128, 256]
       ↓
  ┌──────────────────────────────────────────────────┐
  │ ConvNeXt Backbone (_en)                          │
  │                                                  │
  │  Stem: Conv3×3 s2 → GELU → DWConv3×3 s2 → GELU │
  │        → Conv1×1 → GELU       [1,64,32,64]      │
  │                                                  │
  │  Stage 0: 64ch,  2 blocks     [1,64,32,64]      │
  │  Stage 1: 128ch, 2 blocks     [1,128,16,32]     │
  │  Stage 2: 256ch, 6 blocks     [1,256,8,16]      │
  │  Stage 3: 512ch, 2 blocks     [1,512,4,8]       │
  │                                                  │
  │  Final Conv: GConv3×3(g=512,512→1024)→SE→GELU   │
  │  Head: GlobalAvgPool → FC(1024→2048)             │
  └──────────────────┬───────────────────────────────┘
                     ↓ 2048D
  ┌──────────────────┼──────────────────────────────┐
  │                  │                              │
  ↓                  ↓                              ↓
policy           no_bottleneck                 summarizer
Summarizer       Summarizer                    (2048→512)
(2048→512)       (2048→512)                    L2 Norm
  ↓                  ↓                              ↓
Hydra ResBlock   Hydra ResBlock                hidden_state
  ↓                  ↓                          = 512D
  ↓                  ↓
┌─────────┐    ┌───────────┐
│lead 144 │    │meta    55 │
│lead_p  3│    │desire  32 │
│ll_prob  8│    │road_t  12│
│r_edges264│    │pose    12│
│l_lines528│    │w_euler  6│
└─────────┘    └───────────┘
  = 947D          = 117D

Output: Concat → [1, 1576]  (947 + 117 + 512)
```

**关键架构特征**：

1. **双目输入融合**：窄角 `img`（fcam）和广角 `big_img`（ecam）在通道维直接 Concat（24ch），共享同一个 backbone。**卡车方案只使用 fcam 的 12ch 输入**，Stem 层输入通道从 24 改为 12
2. **ConvNeXt Block 设计**：每个 block 包含**双尺度串行深度卷积**（3×3 DWConv → 7×7 DWConv，串行无激活），残差连接**仅包裹 MLP 部分**（不含 token_mixer），这是 openpilot 对标准 ConvNeXt 的自定义改进
3. **SE 注意力**：仅在 final_conv 处使用一个 Squeeze-and-Excitation block（1024→64→1024）。Final Conv 是分组卷积（g=512, 512→1024 通道翻倍），非标准 DWConv
4. **Summarizer + Hydra 输出头**：2048D 特征经过独立的 Summarizer（FC 2048→512 + ResBlock×2）压缩后，再分支到多个 Hydra Head。其中 policy 和 hidden_state Summarizer 末尾有 L2 Norm，no_bottleneck Summarizer 无 L2 Norm
5. **无 BatchNorm/LayerNorm**：backbone 架构上不使用任何归一化层（非推理时折叠，而是设计上以 LayerScale 替代），backbone 使用 GELU 激活，输出头使用 ReLU
6. **全 float16**：所有参数 float16 存储

### 4.2 卡车模型架构设计

#### 4.2.1 从 openpilot 视觉网络简化

卡车模型基于 openpilot 视觉网络架构进行**针对性简化**，而非从零设计。采用**单目窄角相机（fcam）**作为唯一视觉输入：

```
Input: fcam [1,12,128,256] (单目窄角)  ← 纯图像输入，无需高度参数
       ↓ Cast → [1, 12, 128, 256]
       ↓
  ┌──────────────────────────────────────────────────┐
  │ ConvNeXt Backbone (精简版)                        │
  │                                                  │
  │  与 openpilot 相同的 Stem + 4 Stage 结构          │
  │  可选：减少 Stage 2 的 block 数(6→3)降低计算量    │
  │  可选：通道数减半(64→32,128→64,256→128,512→256)  │
  │                                                  │
  │  Final Conv + SE → GlobalAvgPool → FC → 1024D    │
  └──────────────────┬───────────────────────────────┘
                     ↓
              Summarizer (1024→512)
              + ResBlock + ReLU
                     ↓
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │Lane  │Lead  │Meta  │Pose  │Road  │Road  │Desire│
  │Lines │      │(FCW) │      │Trans │Edges │Pred  │
  │ 528  │ 144  │ 10   │ 12   │ 12   │ 264  │  32  │
  └──────┘──────┘──────┘──────┘  ↑   └──────┘──────┘
                                 │
                          z 分量 = 相机高度估计
                          (供 calibrationd 在线更新)
```

**单目 fcam 设计依据**：
- openpilot 的 `img` 输入对应窄角 fcam（focal length 2648，~40° FOV），`big_img` 对应广角 ecam（focal length 567，~120° FOV）
- 两路输入在 backbone 入口 Concat 为 24ch，共享特征提取。卡车方案去掉 ecam，只保留 fcam 的 12ch 输入
- fcam 的高焦距在远距离（60-100m+）提供更高的像素分辨率，是 LDW 车道线检测和 FCW 前车测距的核心输入
- ecam 的广角能力主要服务于近距离侧向感知和弯道场景，在卡车 LDW/FCW 告警场景中优先级较低
- 后续如需扩展，可增加 ecam 输入，将 12ch 扩展回 24ch，backbone 其余部分无需修改

**与原始视觉网络的差异**：

| 方面 | openpilot 视觉网络 | 卡车精简版 |
|------|---------------------|-----------|
| 输入 | 双目 24ch (fcam + ecam) | **单目 fcam 12ch** |
| 相机焦距 | fcam 2648 + ecam 567 | fcam 2648 |
| Backbone 通道 | 64/128/256/512 | 可减半: 32/64/128/256 |
| Stage 2 block 数 | 6 | 可减至 3 |
| FC Head 输出 | 2048D | 1024D |
| 输出头数 | 3 路 (policy + no_bottleneck + summarizer) | 1 路 (合并的 Summarizer + Hydra) |
| hidden_state | 512D (传递给策略网络) | 移除（无策略网络） |
| wide_from_device_euler | 6D | 移除（无 ecam） |
| meta | 55D (全部事件) | 10D (仅 HARD_BRAKE_3/5) |
| 总输出 | 1576D | ~1013D |
| 估计参数量 | 23M | ~5-8M |

#### 4.2.2 Backbone 规模选项

基于 openpilot ConvNeXt 架构的三种规模方案：

| 方案 | 通道配置 | Stage 2 blocks | 估计参数量 | 适用场景 |
|------|----------|----------------|-----------|---------|
| **Full**（直接沿用） | 64/128/256/512 | 6 | ~15M | 精度优先，GPU 推理 |
| **Medium**（推荐起步） | 48/96/192/384 | 4 | ~8M | 精度与效率平衡 |
| **Small**（NPU 部署） | 32/64/128/256 | 3 | ~3M | NPU 部署，INT8 后 < 3MB |

**建议策略**：
1. 起步使用 **Full** 方案训练，确保感知精度达标
2. 验证通过后，通过**知识蒸馏**将 Full → Small，用于 NPU 部署
3. 如 Small 方案精度不足，可退回 Medium 方案

#### 4.2.3 ConvNeXt Block 详细结构

openpilot 的 ConvNeXt Block 与标准 ConvNeXt 的区别在于**双尺度串行深度卷积**和**残差仅包裹 MLP**设计：

```python
# openpilot ConvNeXt Block（从 ONNX 逆向，节点 [47-65]）
class ConvNeXtBlock(nn.Module):
    """双尺度串行深度卷积 ConvNeXt Block
    关键特征：
    1. 3×3 和 7×7 DWConv 串行连接（非并行相加）
    2. 残差连接仅包裹 MLP 部分（不含 token_mixer）
    3. 无 BatchNorm/LayerNorm
    """
    def __init__(self, dim, expand_ratio=3):
        super().__init__()
        # Token Mixer: 3×3 DWConv（局部特征混合）
        self.token_mixer = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # MLP: 7×7 DWConv → 1×1 expand → GELU → 1×1 project
        self.mlp_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # 大感受野
        self.fc1 = nn.Conv2d(dim, dim * expand_ratio, 1)  # expand
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(dim * expand_ratio, dim, 1)  # project

        # LayerScale (可学习的残差缩放)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)

    def forward(self, x):
        # Token Mixer（串行，无激活）
        x = self.token_mixer(x)          # DWConv 3×3
        # MLP（残差仅包裹此部分）
        h = self.mlp_conv(x)             # DWConv 7×7
        h = self.act(self.fc1(h))        # expand + GELU
        h = self.fc2(h)                  # project
        h = h * self.layer_scale         # LayerScale
        return x + h                     # 残差: token_mixer 输出 + MLP 输出
```

#### 4.2.4 输入格式

**卡车模型输入**（单目 fcam）：
- 形状：`[1, 12, 128, 256]`，uint8
- 12 通道 = 2 帧 × 6 通道（YUV420 拆分为通道维度）
- 空间分辨率 128×256 对应 `MEDMODEL_INPUT_SIZE = (512, 256)` 经 YUV420 转换后的结果
- fcam 图像经 `get_warp_matrix()` 校正后裁剪到模型输入分辨率

**与 openpilot 的输入对比**：
- openpilot：`img [1,12,128,256]`（fcam）+ `big_img [1,12,128,256]`（ecam）→ Concat → `[1,24,128,256]`
- 卡车：仅 `img [1,12,128,256]`（fcam），无 ecam 输入
- Warp 变换使用 fcam 的内参（`camera.py` 中 `fcam.intrinsics`，focal=2648）计算，与 openpilot 的 `road-only` 模式一致

**相机内参选择**：
- 模型坐标系使用 `medmodel_intrinsics`（`model.py:15-18`，focal=910.0，cy=47.6）
- 物理相机使用 fcam 内参（focal=2648）
- `get_warp_matrix()` 在两者之间建立映射，与相机高度无关（仅补偿 RPY 旋转）

### 4.3 高度自估计（模型从图像推断安装高度）

#### 4.3.1 设计思路

openpilot 的视觉网络**已经具备从图像推断相机安装高度的能力**——通过 `road_transform` 输出的 z 分量。这一机制无需任何显式高度输入，网络从图像中的视觉线索自行推断：

- **地面消失点位置**：安装越高，消失点在图像中越高
- **地平面透视几何**：已知宽度的车道标线在不同高度下投影尺度不同
- **已知尺寸参照物**：车辆、标线宽度等提供绝对尺度参考

因此，卡车模型**不需要外部注入高度条件**（如 FiLM 层），而是沿用 openpilot 的设计，让模型自行估计高度。

#### 4.3.2 与 openpilot 现有机制的对应关系

openpilot 的高度估计数据流：

```
视觉网络 → road_transform[0:3] (平移分量)
                    ↓
         fill_pose_msg (modeld.py)
                    ↓
         cameraOdometry.roadTransformTrans
                    ↓
         calibrationd.handle_cam_odom()
                    ↓
         road_transform_trans[2] → new_height    # z 分量 = 相机高度
                    ↓
         滑动平均 → liveCalibration.height
```

源码关键路径：
- 视觉网络输出 `road_transform`（`parse_model_outputs.py:101`）
- 发布到 `cameraOdometry.roadTransformTrans`（`fill_model_msg.py:189`）
- `calibrationd` 从 z 分量提取高度（`calibrationd.py:259-260`）：
  ```python
  if (len(road_transform_trans) == 3):
      new_height = np.array([road_transform_trans[2]])
  ```
- 通过滑动平均平滑后发布（`calibrationd.py:268`）

#### 4.3.3 模型自估计的优势

| 维度 | 外部高度注入（FiLM） | 模型自行估计 |
|------|---------------------|-------------|
| 架构复杂度 | 增加 FiLM 层 + HeightEncoder | 无额外模块，更简洁 |
| 推理时输入 | 需要提供高度标量 | 只需图像 |
| 载重动态变化 | 需外部高度源实时更新 | 模型逐帧自适应，天然处理空载/满载 |
| 与 openpilot 一致性 | 偏离原有设计 | **完全一致**，复用全部标定流程 |
| 已验证性 | 需要从零验证 | openpilot 百万级数据已验证 |
| NPU 部署 | 多一个输入通道，多 FiLM 算子 | 纯 CNN，更友好 |

#### 4.3.4 训练要求

模型自估计高度的前提是**训练数据必须覆盖目标高度范围**。具体策略：

1. **Carla 仿真阶段**：使用 `--camera-height` 参数在 [1.0, 3.0]m 范围内均匀采样，确保每个高度档位有足够训练样本
2. **GT 标签**：`road_transform` 的 z 分量 GT 直接取自 Carla 的相机安装高度参数
3. **验证指标**：在不同高度上分别评估 `road_transform.z` 的 MAE，确保高度估计误差 < 0.1m

如果后续验证发现多高度泛化不足（例如车道线精度在极端高度下显著退化），可考虑回退到 FiLM 条件化方案作为补救。但基于 openpilot 已有的成功经验，模型自估计应是优先选择。

### 4.4 时序建模

**openpilot 实际方案（从 ONNX 确认）**：直接将 2 帧在通道维拼接为 12ch（每帧 6ch YUV），输入同一个 backbone。无 ConvGRU、无显式时序模块——时序信息完全通过多帧通道拼接隐式编码。

卡车模型**直接沿用此方案**（单目 fcam 双帧拼接）：

```
fcam_t-1 (6ch YUV) + fcam_t (6ch YUV) → concat → [B, 12, 128, 256] → backbone
```

这是最简单高效的时序融合方式，且已被 openpilot 验证有效。网络通过学习帧间差异隐式获取运动信息（速度、光流等）。

**策略网络的时序机制不再需要**：openpilot 策略网络通过 `hidden_state`（512D）在 25 帧间传递（GPT 风格 Transformer + `features_buffer [1,25,512]`），服务于控制决策的时序连贯性。卡车告警系统不需要策略网络，因此这套 Transformer 时序机制整体移除，每帧独立推理即可。

### 4.5 输出头设计

#### 4.5.1 openpilot 实际输出头结构（ONNX 分析）

openpilot 视觉网络的输出头采用 **Summarizer + Hydra** 模式，而非简单的独立 FC：

```python
# 每个输出头的实际结构（从 ONNX 逆向）
class HydraHead(nn.Module):
    """Hydra 输出头：投影 → ResBlock → 最终投影"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)  # 投影到 head 内部维度
        self.res_block_a = nn.Sequential(               # ResBlock
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.res_block_b = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = x + self.res_block_a(x)   # 残差
        x = x + self.res_block_b(x)   # 残差
        return self.final_layer(F.relu(x))
```

```python
# Summarizer：压缩 backbone 输出到固定维度（从 ONNX 逆向）
class Summarizer(nn.Module):
    """2048D → 512D 压缩器
    注意：policy 和 hidden_state 末尾有 FC + L2 Norm，no_bottleneck 无 L2 Norm
    """
    def __init__(self, has_l2_norm=True):
        super().__init__()
        self.fc_in = nn.Linear(2048, 512)
        # 2 层 ResBlock (512→1024→512)
        self.res_block_1 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512))
        self.res_block_2 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512))
        self.has_l2_norm = has_l2_norm
        if has_l2_norm:
            self.fc_mu = nn.Linear(512, 512)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = F.relu(x + self.res_block_1(F.relu(x)))  # ResBlock 1
        x = F.relu(x + self.res_block_2(F.relu(x)))  # ResBlock 2
        if self.has_l2_norm:
            x = self.fc_mu(F.relu(x))
            return F.normalize(x, dim=-1)  # L2 归一化
        return x  # no_bottleneck 直接输出
```

openpilot 将输出头分为三路（共享 backbone 2048D 特征）：

| 路径 | Summarizer 维度 | L2 Norm | Hydra Heads | 总输出 |
|------|-----------------|---------|-------------|--------|
| `policy` | 2048→512 | **有** | lead(144), lead_prob(3), lane_lines_prob(8), road_edges(264), lane_lines(528) | 947D |
| `no_bottleneck` | 2048→512 | **无** | meta(55), desire_pred(32), road_transform(12), pose(12), wide_from_device_euler(6) | 117D |
| `hidden_state` | 2048→512 | **有** | — | 512D (hidden_state) |

每个 Hydra Head 的内部隐藏维度（从 ONNX 确认）：

| Head | in_dim → hidden_dim → out_dim |
|------|-------------------------------|
| lead | 512 → 64 → 144 |
| lead_prob | 512 → 16 → 3 |
| lane_lines_prob | 512 → 16 → 8 |
| road_edges | 512 → 32 → 264 |
| lane_lines | 512 → 64 → 528 |
| meta | 512 → 64 → 55 |
| desire_pred | 512 → 32 → 32 |
| road_transform | 512 → 32 → 12 |
| pose | 512 → 32 → 12 |
| wide_from_device_euler | 512 → 32 → 6 |

#### 4.5.2 卡车模型输出头设计

沿用 Summarizer + Hydra 模式，合并为**单路**（无需三路分离，因为不需要 hidden_state 传递给策略网络）：

| Head | hidden_dim | 输出维度 | 解析方式 | 说明 |
|------|------------|----------|----------|------|
| lane_lines | 64 | 528 | MDN (mean + log_std) | 4 条车道线 y/z 坐标 |
| lane_lines_prob | 16 | 8 | BCE sigmoid | 车道线存在概率 |
| lead | 64 | 144 | MDN | 前车轨迹 (x,y,v,a) |
| lead_prob | 16 | 3 | BCE sigmoid | 前车存在概率 |
| meta_fcw | 32 | 10 | BCE sigmoid | HARD_BRAKE_3/5 各 5 时间点 |
| pose | 32 | 12 | MDN | 视觉里程计 |
| road_transform | 32 | 12 | MDN | 路面几何/高度估计 |
| road_edges | 32 | 264 | MDN | 路缘线 |
| desire_pred | 32 | 32 | CE softmax | 意图预测 |

---

## 第5章：训练策略

### 5.1 三阶段训练总览

| 阶段 | 训练数据 | 模型规模 | 训练目标 | 硬件 |
|------|----------|----------|----------|------|
| Phase 1 | Carla 仿真 | **Full**（~15M params） | 多高度精度验证 | RTX 4090 GPU |
| Phase 2 | Carla + openpilot 数据 | **Small**（~3M params） | 双 Teacher 蒸馏压缩 | RTX 4090 GPU |
| Phase 3 | 真实卡车数据 | Small（微调） | 真实场景适配 | RTX 4090 GPU |

### 5.2 Phase 1：Carla 仿真训练 — Full 模型（0-4 月）

#### 5.2.1 目标

在 Carla 仿真数据上训练一个 **Full 规模**（通道 64/128/256/512，Stage 2 共 6 blocks，~15M params）的卡车视觉模型，侧重**多种安装高度场景的精度**。此阶段的核心任务是验证网络架构在卡车高度范围（1.2-2.8m）下的感知能力上限。

#### 5.2.2 训练平台

- GPU：NVIDIA RTX 4090（24GB VRAM）
- Full 模型 FP32 训练预估显存：~8-12GB（batch size 16-32）
- 训练框架：PyTorch

#### 5.2.3 数据采集

利用已有的 `tools/dashcam/` 基础设施，数据采集管线已基本就绪。

##### 采集能力总览

**Carla 仿真管理**（`carla_world.py` — `DashcamCarlaWorld` 类）：
- 同步仿真模式，固定时间步 0.025s（40 Hz tick），相机传感器 20 FPS
- 相机分辨率 1928×1208（标准 openpilot 格式），窄角 FOV=40°（`--road-only`）或双目 40°+120°
- Ego 车辆动态速度控制：`--speed-range MIN MAX`（默认 20~140 km/h），每 8~20 秒随机切换目标速度，TM 自动处理加减速过渡，产生多样化的速度场景
- `--random-spawn` 随机出生点，`--num-npc` NPC 车辆数量控制

**训练数据录制**（`run.py` + `data_recorder.py`）：
- `--record <dir>` 启用录制，自动启用 `--road-only` 和 `--fast`（跳过帧率限制，最大化采集速度）
- `--record-only` 纯录制模式：跳过 modeld 和可视化，仅采集 GT 数据，吞吐量最高
- `--record-skip N` 帧跳跃采样（默认 1 = 每帧保存）
- 每帧保存为独立 NPZ 压缩文件（`000001.npz`, `000002.npz`, ...），附带 `clip_info.json` 剪辑元数据

**Ground Truth 提取器**：
- `lane_ground_truth.py`：从 Carla HD Map 提取 4 条车道线 + 2 条路沿的精确 3D 坐标，在 X_IDXS（33 个前向距离点）处样条插值，含山顶截断处理
- `lead_ground_truth.py`：从 Carla actor 列表提取前车 GT，3 个时间偏移选择（0/2/4s）× 6 个预测时间步（0~10s）× 4 维状态（x, y, v, a），使用匀加速模型预测未来轨迹
- `pose_ground_truth.py`：从 Carla ego 变换计算帧间速度和角速度（校准坐标系）
- `lane_evaluator.py`：车道线检测精度在线评估

**验证与可视化工具**：
- `view_npz.py`：加载 NPZ，warp 到 512×256 模型输入空间，叠加车道线/路沿/前车标注，支持交互浏览和批量导出 PNG
- `verify_pose.py`：验证 pose 与 v_ego 一致性、轨迹积分精度、校准坐标系正确性
- `warp_example.py`：演示 rpyCalib 对图像 warp 的影响

##### NPZ 训练数据格式

每帧 NPZ 文件包含以下字段：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `frame_rgb` | [1208, 1928, 3] | uint8 | 原始 RGB 图像 |
| `lane_lines` | [4, 33, 3] | float32 | 车道线 GT：4 条线 × 33 个 X_IDXS 距离 × (x, y, z) |
| `lane_lines_prob` | [4] | float32 | 车道线存在概率（0.0 或 1.0） |
| `road_edges` | [2, 33, 3] | float32 | 路沿 GT：2 条边 × 33 点 × (x, y, z) |
| `road_edges_prob` | [2] | float32 | 路沿存在概率 |
| `lead` | [3, 6, 4] | float32 | 前车 GT：3 选择 × 6 时间步 × (x_dist, y_offset, v_abs, accel) |
| `lead_prob` | [3] | float32 | 前车存在概率 |
| `pose` | [6] | float32 | 帧间运动：[v_fwd, v_lat, v_vert, w_roll, w_pitch, w_yaw] (m/s, rad/s) |
| `road_transform` | [6] | float32 | 路面变换：[0, 0, camera_height, 0, 0, 0] |
| `rpyCalib` | [3] | float32 | 每帧标定角 [roll, -(pitch+车辆俯仰), -yaw]，单位 rad |
| `v_ego` | scalar | float32 | 车速 (m/s) |
| `world_pose` | [6] | float32 | Carla 世界坐标 [x, y, z, roll°, pitch°, yaw°] |
| `camera_height` | scalar | float32 | 相机安装高度 (m) |
| `camera_pitch` | scalar | float32 | 相机安装俯仰角 (rad) |
| `camera_yaw` | scalar | float32 | 相机安装偏航角 (rad) |
| `town` | scalar | str | Carla 地图名 |

> **GT 与模型输出的维度差异**：GT 中 `lane_lines` 为 [4, 33, 3]（含 x 坐标），`road_edges` 为 [2, 33, 3]；而模型输出仅为 [4, 33, 2]（y, z）和 [2, 33, 2]（y, z），x 维度来自固定的 `X_IDXS`。训练时需对齐：GT 的 x 维度可用于验证采样点正确性，损失函数仅计算 y/z 分量。

##### 典型采集命令

```bash
# 单次录制（窄角相机，高度 2.4m，理想安装）
python tools/dashcam/run.py --record data/train/town04_h2.4 \
  --record-only --camera-height 2.4 --perfect-cam --max-frames 2000

# 指定速度范围和 NPC 数
python tools/dashcam/run.py --record data/train/town04_slow \
  --record-only --camera-height 2.0 --speed-range 20 60 --num-npc 30

# 随机出生点，全速度范围
python tools/dashcam/run.py --record data/train/town04_rand \
  --record-only --random-spawn --camera-height 1.8 --max-frames 5000

# 验证录制数据
python tools/dashcam/view_npz.py data/train/town04_h2.4/
python tools/dashcam/verify_pose.py data/train/town04_h2.4/ --plot
```

##### 数据量目标

~50k 帧，覆盖以下维度：
- **高度**：1.2m, 1.8m, 2.0m, 2.4m, 2.8m（5 个档位，均匀采样）
- **速度**：20~140 km/h 动态变化（默认 `--speed-range`），涵盖低速、巡航、高速场景
- **Town**：Town01-Town07（不同道路结构）
- **天气**：晴天/阴天/雨天/雾天
- **时段**：白天/黄昏/夜间

#### 5.2.4 训练重点：多高度精度

Phase 1 的核心挑战是让模型在 1.2-2.8m 的高度范围内均具备良好的感知精度。策略如下：

1. **高度均衡采样**：每个 batch 中各高度档位的样本数量大致均等，防止模型偏向某一高度
2. **高度感知数据增强**：在 [1.0, 3.0]m 范围内连续随机采样高度（不仅限于 5 个离散档位），提升模型对中间高度的泛化
3. **分高度评估**：训练过程中按高度分组评估 LDW/FCW 指标，确保各高度段精度均衡：
   - 乘用车高度（1.2m）：验证与 openpilot 基线的可比性
   - 卡车常见高度（2.0-2.4m）：核心目标精度
   - 极端高度（2.8m）：确认无严重退化
4. **`road_transform.z` 高度估计精度**：作为辅助指标，各高度段的 MAE 应 < 0.1m

**Phase 1 达标标准**：

| 指标 | 目标 | 评估数据 |
|------|------|----------|
| LDW 车道线 y MAE (0-60m) | < 0.3m（各高度段均满足） | Carla 验证集 |
| FCW lead dRel 误差 | < 15% (0-100m) | Carla 验证集 |
| road_transform.z MAE | < 0.1m（各高度段） | Carla 验证集 |
| 高度间精度方差 | 各高度段 MAE 差异 < 0.1m | Carla 验证集 |

#### 5.2.5 其他数据增强

| 增强类型 | 方法 | 目的 |
|----------|------|------|
| 几何 | 随机 pitch/yaw/roll 偏移 | 模拟安装偏差 |
| 高度 | 在 [1.0, 3.0]m 连续采样 | 覆盖高度范围 |
| 光照 | 亮度/对比度/色彩抖动 | 应对不同光照 |
| 遮挡 | 随机矩形遮挡 | 增强鲁棒性 |
| 模糊 | 运动模糊/高斯模糊 | 模拟真实退化 |

### 5.3 Phase 2：双 Teacher 蒸馏 — Small 模型（4-6 月）

#### 5.3.1 目标

使用**两个 Teacher** 联合蒸馏，训练一个可部署到 NPU 的 **Small 规模**模型（通道 32/64/128/256，Stage 2 共 3 blocks，~3M params）。

#### 5.3.2 双 Teacher 架构

```
  Teacher A                    Teacher B
  (Phase 1 Full 模型,          (openpilot supercombo,
   ~15M, frozen)                23M 视觉网络, frozen)
      ↓                             ↓
   多高度场景                    乘用车高度场景
   soft labels                   soft labels
      ↓                             ↓
      └──────────┬──────────────────┘
                 ↓
         Student (Small, ~3M)
              ↓
         L_total = α·L_gt + β·L_distill_A + γ·L_distill_B + δ·L_temporal
```

#### 5.3.3 双 Teacher 蒸馏策略

**Teacher A — Phase 1 Full 模型**：
- 在**全部高度**的 Carla 数据上提供 soft labels
- 优势：已在卡车高度范围（1.2-2.8m）上验证精度，多高度知识最全面
- 权重 β 在全部训练数据上均生效

**Teacher B — openpilot supercombo 视觉网络**：
- 仅在**乘用车高度**（~1.22m）的数据上提供 soft labels
- 优势：在百万级真实驾驶数据上训练，视觉特征提取和场景理解能力极强
- 权重 γ 仅在 1.22m 高度数据上生效（其他高度不使用 Teacher B 的输出）

**为什么需要两个 Teacher**：
- Teacher A（Full 模型）提供**多高度泛化能力**，但仅在 Carla 仿真数据上训练，真实场景泛化有限
- Teacher B（supercombo）提供**真实场景理解能力**，但仅在乘用车高度有效
- 双 Teacher 互补：Student 同时继承 Full 模型的高度泛化和 supercombo 的真实场景鲁棒性

#### 5.3.4 蒸馏损失设计

```python
# 总损失
L_total = α * L_gt + β * L_distill_A + γ * L_distill_B + δ * L_temporal

# Teacher A 蒸馏（所有高度数据）
L_distill_A = distillation_loss(student_out, teacher_A_out)

# Teacher B 蒸馏（仅 h ≈ 1.22m 的数据）
L_distill_B = distillation_loss(student_out, teacher_B_out) * mask_car_height

# mask_car_height: 当前样本高度 ∈ [1.1, 1.4]m 时为 1，否则为 0
```

**权重调度建议**：
- 训练初期：β=1.0, γ=1.0, α=0.5（以蒸馏为主，GT 为辅）
- 训练中期：β=0.5, γ=0.5, α=1.0（逐渐转向 GT 监督）
- 训练后期：β=0.3, γ=0.3, α=1.0（微调，减少蒸馏依赖）

#### 5.3.5 输出对齐

Teacher A 和 Student 输出结构完全一致（同为卡车精简版 ~1013D），可直接逐 head 对齐。

Teacher B（openpilot supercombo）的输出维度更大（1576D），需要选择性对齐：
- `lane_lines`（528D）、`lane_lines_prob`（8D）：直接对齐
- `lead`（144D）、`lead_prob`（3D）：直接对齐
- `pose`（12D）、`road_transform`（12D）、`road_edges`（264D）：直接对齐
- `meta`：Teacher B 输出 55D，Student 只需 10D（HARD_BRAKE_3/5），取对应 slice 对齐
- `desire_pred`（32D）：直接对齐
- `hidden_state`（512D）、`wide_from_device_euler`（6D）：Student 无对应输出，跳过

#### 5.3.6 Phase 2 达标标准

| 指标 | 目标 | 说明 |
|------|------|------|
| LDW 车道线 y MAE (0-60m) | < 0.35m（各高度段） | 允许比 Full 略有退化 |
| FCW lead dRel 误差 | < 18% (0-100m) | 允许比 Full 略有退化 |
| 相对 Teacher A 精度保持率 | > 90% | Student 精度 / Teacher A 精度 |
| 模型参数量 | ~3M | INT8 后 < 5MB |

### 5.4 Phase 3：真实卡车数据微调（6-9 月）

#### 5.4.1 目标

在 Phase 2 得到的 Small 模型基础上，使用真实卡车采集数据微调，弥合仿真-真实域差距（sim-to-real gap）。

#### 5.4.2 数据采集与标注

- 在实际卡车上安装 fcam 窄角相机采集道路数据
- **自监督标注**：参考 openpilot 的 future-retrospective 方法：
  - 利用后续帧的视觉里程计（`pose` 输出）将未来观测投影回当前帧
  - 多帧时序累积提高远距离车道线精度
  - 前车检测可用成熟的 2D 检测器（如 YOLO）在图像上标注，再投影到 3D
- **半监督学习**：少量人工标注 + 大量自监督/伪标签数据

#### 5.4.3 微调策略

- 使用较小学习率（Phase 2 的 1/10），防止遗忘 Carla/蒸馏阶段学到的知识
- 混合训练：真实数据 + 少量 Carla 数据（防止灾难性遗忘）
- 重点关注 sim-to-real 差异较大的场景：光照变化、道路纹理、标线磨损、天气恶劣

#### 5.4.4 Phase 3 达标标准

| 指标 | 目标 | 评估数据 |
|------|------|----------|
| LDW 车道线 y MAE (0-60m) | < 0.5m | 真实卡车验证集 |
| FCW lead dRel 误差 | < 15% (0-100m) | 真实卡车验证集 |
| LDW 误报率 | < 1次/100km | 真实路测 |
| FCW 漏报率 | < 5% | 真实路测 |

---

## 第6章：损失函数设计

### 6.1 总体损失

**Phase 1（Full 模型，纯 GT 监督）**：
```
L_total = L_gt + δ·L_temporal
```

**Phase 2（Small 模型，双 Teacher 蒸馏）**：
```
L_total = α·L_gt + β·L_distill_A(Full) + γ·L_distill_B(supercombo) + δ·L_temporal
```

**Phase 3（Small 模型微调）**：
```
L_total = L_gt + δ·L_temporal
```

各权重在 Phase 2 训练过程中逐步调整（详见第 5.3.4 节）。

### 6.2 各任务损失详解

#### 车道线：Laplace NLL

openpilot 模型使用 MDN 输出 (mean, log_std)，对应 Laplace 分布：

```python
def laplace_nll_loss(pred_mean, pred_log_std, target):
    """Laplace 负对数似然损失
    L = |target - pred_mean| / exp(pred_log_std) + pred_log_std
    """
    b = torch.exp(pred_log_std).clamp(min=1e-6)
    return (torch.abs(target - pred_mean) / b + pred_log_std).mean()

# 对 lane_lines 的 y 和 z 分别计算
L_lane_y = laplace_nll_loss(pred_lane_y, pred_lane_y_logstd, gt_lane_y)
L_lane_z = laplace_nll_loss(pred_lane_z, pred_lane_z_logstd, gt_lane_z)
L_lane = L_lane_y + L_lane_z
```

车道线概率使用 BCE：
```python
L_lane_prob = F.binary_cross_entropy_with_logits(pred_lane_logits, gt_lane_visible)
```

#### 前车：Laplace NLL + 概率 BCE

```python
# 前车轨迹 NLL
L_lead_traj = laplace_nll_loss(pred_lead, pred_lead_logstd, gt_lead)
# 前车存在概率 BCE
L_lead_prob = F.binary_cross_entropy_with_logits(pred_lead_prob_logits, gt_lead_exists)
L_lead = L_lead_traj + L_lead_prob
```

#### FCW 急刹预测：BCE

```python
# meta 中 HARD_BRAKE_3/5 的概率预测
L_meta = F.binary_cross_entropy_with_logits(pred_brake_logits, gt_brake_labels)
```

GT 构造方式：基于未来 2/4/6/8/10 秒的实际减速度，标注是否发生了 -3m/s² 或 -5m/s² 的急刹。

#### 视觉里程计：Laplace NLL + uncertainty weighting

```python
# pose: trans(3D) + rot(3D)
L_pose = laplace_nll_loss(pred_pose, pred_pose_logstd, gt_pose)
```

#### 时序一致性正则

```python
# 当前帧特征与上一帧特征经 ego-motion warp 后的一致性
L_temporal = ||f(t) - warp(f(t-1), ego_motion)||_1
```

鼓励网络学习在时序上一致的表征，减少检测结果的帧间跳动。

### 6.3 蒸馏损失（Phase 2 使用）

```python
def distillation_loss(student_out, teacher_out, temperature=3.0):
    """单个 Teacher 的蒸馏损失"""
    # 对 MDN 输出：L2 距离（mean）+ KL（分布）
    L_mean = F.mse_loss(student_out['mean'], teacher_out['mean'])
    L_std = F.mse_loss(student_out['log_std'], teacher_out['log_std'])
    # 对概率输出：KL 散度
    L_prob = F.kl_div(
        F.log_softmax(student_out['logits'] / temperature, dim=-1),
        F.softmax(teacher_out['logits'] / temperature, dim=-1),
        reduction='batchmean') * (temperature ** 2)
    return L_mean + L_std + L_prob

def dual_teacher_distillation(student_out, teacher_A_out, teacher_B_out,
                               sample_height, temperature=3.0):
    """双 Teacher 蒸馏损失
    Teacher A: Phase 1 Full 模型 — 所有高度数据上生效
    Teacher B: openpilot supercombo — 仅乘用车高度数据上生效
    """
    L_A = distillation_loss(student_out, teacher_A_out, temperature)

    # Teacher B 仅在乘用车高度 (1.1-1.4m) 上提供监督
    car_height_mask = (sample_height > 1.1) & (sample_height < 1.4)
    L_B = distillation_loss(student_out, teacher_B_out, temperature)
    L_B = L_B * car_height_mask.float()

    return L_A, L_B
```

---

## 第7章：相机标定与高度估计

### 7.1 在线标定复用

直接复用 openpilot 的 `calibrationd`（`selfdrive/locationd/calibrationd.py`），核心修改：

**修改 `HEIGHT_INIT` 默认值**（`calibrationd.py:52`）：
```python
# 原始值（乘用车）
HEIGHT_INIT = np.array([1.22])

# 卡车默认值
HEIGHT_INIT = np.array([2.40])  # 根据具体卡车型号调整
```

**RPY 标定逻辑无需修改**。标定核心算法（`calibrationd.py:212-278`）：

1. 速度门限：`v_ego > MIN_SPEED_FILTER`（`15 mph ≈ 6.7 m/s`）
2. 直线行驶：`abs(rot[2]) < MAX_YAW_RATE_FILTER`（`2°/s`）
3. 从视觉里程计 `trans` 反推俯仰/偏航观测量：
   ```python
   observed_rpy = [0, -arctan2(trans[2], trans[0]), arctan2(trans[1], trans[0])]
   ```
4. 与历史 RPY 平滑融合（block-based sliding window）
5. 当 `valid_blocks >= INPUTS_NEEDED = 5` 时认为标定有效

### 7.2 Warp Matrix

`get_warp_matrix()`（`common/transformations/model.py:65-70`）完全复用：

```python
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix
```

变换链：`model_frame ← calib_frame ← device_frame ← camera_frame`

关键点：
- 只补偿旋转，不含平移/高度
- `medmodel_fl = 910.0`，`MEDMODEL_CY = 47.6`（`model.py:13-14`）
- 高度信息由模型通过 `road_transform` 输出自行估计（而非通过几何变换或外部注入）

### 7.3 高度在线估计

高度更新流程（`calibrationd.py:259-268`）：

```python
# 当 road_transform 足够稳定时，从 z 分量更新高度
if len(road_transform_trans) == 3:
    new_height = np.array([road_transform_trans[2]])
else:
    new_height = HEIGHT_INIT

# 写入滑动平均缓存
self.heights[self.block_idx] = moving_avg_with_linear_decay(
    self.heights[self.block_idx], new_height, self.idx, float(BLOCK_SIZE))
```

**卡车特殊考虑**：
- 初始值来自安装参数（已知的卡车相机高度，如 2.4m）
- 载重变化导致车身高度变化（空载 vs 满载差异 10-20cm）
- `road_transform` 的 z 分量可以在线追踪这种变化
- 高度稳定性阈值 `MAX_HEIGHT_STD = exp(-3.5) ≈ 0.03`（`calibrationd.py:41`）对卡车同样适用

### 7.4 卡车标定参数汇总

| 参数 | 乘用车默认值 | 卡车建议值 | 来源 |
|------|-------------|------------|------|
| `HEIGHT_INIT` | 1.22m | 2.40m | `calibrationd.py:52` |
| `PITCH_LIMITS` | [-0.091, 0.17] rad | 需扩展 | `calibrationd.py:56-58` |
| `YAW_LIMITS` | [-0.069, 0.069] rad | 同 | `calibrationd.py:59` |
| `MIN_SPEED_FILTER` | 15 mph | 可降低到 10 mph | `calibrationd.py:37` |
| `CAMERA_OFFSET` | 0.04m | 需按卡车实测 | `ldw.py:6` |

---

## 第8章：模型部署方案

### 8.1 量化流程

```
FP32 训练 → PTQ (Post-Training Quantization) INT8 → QAT (量化感知训练) → 导出
                                                                          ↓
                                                                 ONNX / TFLite / RKNN
```

**PTQ 步骤**：
1. 在验证集上运行校准（通常需要 200-500 个 batch）
2. 收集每层的激活值范围
3. 选择对称/非对称量化方案
4. 导出 INT8 模型并评估精度损失

**QAT 步骤**（如果 PTQ 精度损失过大）：
1. 在模型中插入伪量化节点
2. 使用训练集继续微调 5-10 个 epoch
3. 学习率设为原训练的 1/10
4. 导出量化后的模型

### 8.2 目标规格

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 模型大小（INT8） | < 5 MB (Small) / < 10 MB (Medium) | NPU 存储约束 |
| 推理延迟（NPU） | < 30 ms @ 20 FPS | 实时性要求 |
| 推理延迟（CPU） | < 100 ms | 降级运行 |
| 输入 | [1, 12, 128, 256] uint8 | 单目 fcam，2 帧 × 6ch YUV |
| 输出维度 | ~1013D | 精简后（去除 hidden_state 和 wide_from_device_euler） |
| 精度要求（LDW） | 车道线 y MAE < 0.3m (0-60m) | 核心指标 |
| 精度要求（FCW） | lead dRel 误差 < 10% (0-100m) | 核心指标 |

### 8.3 运行时架构

```
fcam (窄角相机, focal=2648)
    ↓ YUV420
Warp (calibration RPY) ← liveCalibration (rpyCalib)
    ↓                     (使用 fcam 内参)
Model Input: [1, 12, 128, 256] uint8    ← 单目 fcam，2帧×6ch YUV
    ↓
┌───────────────────────────────────┐
│  Model Inference (NPU/CPU)        │
│  ConvNeXt → Summarizer → Hydra   │
└──────────────┬────────────────────┘
               ↓
         Parse Outputs ──→ road_transform.z ──→ calibrationd
               ↓                                (高度在线更新)
    ┌──────────┴──────────┐
    ↓                     ↓
LDW Logic            FCW Logic
(ldw.py)             ┌────┴────┐
    ↓                ↓         ↓
LDW Alert      hardBrake   TTC Check
               predicted   (dRel/vRel)
                    ↓         ↓
                FCW Alert ←──┘
```

### 8.4 NPU 适配注意事项

1. **算子兼容性**：ConvNeXt 使用的算子（DWConv、GELU、SE Block）需确认 NPU 支持。GELU 的 Tanh 近似可能需要替换为 ReLU/HardSwish
2. **7×7 深度卷积**：部分 NPU 对大 kernel DWConv 支持不佳，可能需要拆分为多个 3×3
3. **内存布局**：NPU 通常偏好 NHWC 或特定对齐的 NCHW
4. **动态 shape**：模型已是全静态 shape（单目 fcam 固定 `[1,12,128,256]`），无需额外处理
5. **后处理**：sigmoid/softmax 等激活函数尽量在 NPU 上完成
6. **单目优势**：相比双目 24ch 输入，单目 fcam 12ch 减少一半的 Stem 计算量和内存带宽

---

## 第9章：LDW/FCW 告警逻辑

### 9.1 LDW 告警

**基础逻辑**：直接复用 `selfdrive/controls/lib/ldw.py` 的判断逻辑。

**卡车适配修改**：

| 参数 | 乘用车值 | 卡车建议值 | 修改理由 |
|------|---------|-----------|----------|
| `CAMERA_OFFSET` | 0.04m | 按实际安装测量 | 卡车相机安装位置不同 |
| 阈值 `1.08m` | 1.08m | 1.25-1.40m | 卡车车身更宽（~2.5m vs ~1.8m） |
| `LDW_MIN_SPEED` | 31 mph ≈ 50 km/h | 40 km/h | 卡车低速场景也需告警 |
| 转向灯冷却 | 5s | 5s | 无需修改 |

**阈值计算依据**：
- 乘用车宽度 ~1.8m，车道宽度 ~3.5m，单侧余量 ~0.85m，阈值 1.08m（略大于余量）
- 卡车宽度 ~2.5m，车道宽度 ~3.75m（货车车道），单侧余量 ~0.625m，阈值可设为 1.25m

### 9.2 FCW 告警

#### 机制1：端到端急刹预测

直接复用 `fill_model_msg.py:143-149` 的逻辑：

```python
# 滑动窗口判定
hard_brake_predicted = (
    (prev_brake_5ms2_probs > [0.05, 0.05, 0.15, 0.15, 0.15]).all() and
    (prev_brake_3ms2_probs > [0.7, 0.7]).all()
)
```

**卡车适配**：可能需要降低阈值（卡车制动响应更慢，需要更早预警）：
```python
# 卡车建议值
FCW_THRESHOLDS_5MS2 = [0.03, 0.03, 0.10, 0.10, 0.10]  # 降低阈值
FCW_THRESHOLDS_3MS2 = [0.5, 0.5]                        # 降低阈值
```

#### 机制2：简化版 TTC 碰撞检测

不再使用 MPC 求解器（太重且为控制设计），改用直接的 TTC 计算：

```python
def compute_ttc(dRel, vRel):
    """计算 Time-to-Collision
    dRel: 与前车纵向距离 (m)
    vRel: 相对速度 (m/s)，接近为负值
    """
    if vRel >= 0:
        return float('inf')  # 前车远离，无碰撞风险
    ttc = -dRel / vRel
    return max(ttc, 0.0)

def check_fcw_ttc(lead_msg, v_ego, model_v_ego):
    """基于纯视觉 lead 信息的 TTC 检查"""
    if lead_msg.prob < 0.5:
        return False

    dRel = lead_msg.x[0] - RADAR_TO_CAMERA  # 1.52m
    vRel = lead_msg.v[0] - model_v_ego

    ttc = compute_ttc(dRel, vRel)

    # 卡车制动距离更长，TTC 阈值更大
    TTC_THRESHOLD = 4.0  # 秒（乘用车通常 2.5-3.0s）

    return ttc < TTC_THRESHOLD and dRel > 0 and v_ego > 5.0  # 最低速度门限
```

**卡车 TTC 阈值选择**：
- 80 km/h 时卡车制动距离 ~70m，减速度约 ~5 m/s²
- 需要 TTC > 4s 才能安全停车
- 建议 TTC 告警阈值：**4.0 秒**（乘用车通常 2.5-3.0 秒）

### 9.3 告警整合

```python
class TruckAlertManager:
    def __init__(self):
        self.ldw = LaneDepartureWarning()  # 复用 ldw.py
        self.fcw_ttc_cnt = 0
        self.prev_brake_5ms2_probs = np.zeros(5)
        self.prev_brake_3ms2_probs = np.zeros(2)

    def update(self, model_output, CS):
        # LDW
        self.ldw.update(frame, modelV2, CS, CC)
        ldw_alert = self.ldw.warning

        # FCW - 机制1：端到端急刹预测
        hard_brake_predicted = self._check_hard_brake(model_output)

        # FCW - 机制2：TTC
        ttc_alert = self._check_ttc(model_output, CS.vEgo)

        fcw_alert = hard_brake_predicted or (self.fcw_ttc_cnt > 2)

        return ldw_alert, fcw_alert
```

---

## 第10章：项目里程碑

| 阶段 | 时间 | 对应训练阶段 | 目标 | 关键交付 |
|------|------|-------------|------|----------|
| **M1: 基础搭建** | 1-2月 | Phase 1 准备 | 数据管线 + Full 网络骨架 | Carla 多高度数据采集脚本；Full 规模 ConvNeXt 网络定义（PyTorch）；RTX 4090 训练框架搭建 |
| **M2: Full 模型训练** | 2-4月 | Phase 1 | Carla 数据训练 Full 模型 | LDW MAE < 0.3m（各高度段）；FCW lead 误差 < 15%；高度估计 MAE < 0.1m |
| **M3: 双 Teacher 蒸馏** | 4-6月 | Phase 2 | Full + supercombo → Small | Small 模型精度保持率 > 90%（相对 Full）；模型参数 ~3M |
| **M4: 真实数据微调** | 6-8月 | Phase 3 | 卡车实采数据微调 Small | LDW MAE < 0.5m（真实场景）；FCW lead 误差 < 15% |
| **M5: 部署优化** | 8-9月 | 部署 | INT8 量化 + NPU 适配 | Small 模型 INT8 < 5MB；NPU 推理 < 30ms；量化精度损失 < 5% |
| **M6: 系统集成** | 9-10月 | 集成 | 完整 LDW/FCW 系统 | 端到端系统测试；误报率 < 1次/100km；漏报率 < 5% |

### 验证标准

| 指标 | M2 (Full) | M3 (Small) | M4 (微调) | M6 (系统) |
|------|-----------|------------|-----------|-----------|
| LDW 车道线 y MAE (0-60m) | < 0.3m (Carla) | < 0.35m (Carla) | < 0.5m (真实) | < 0.5m (真实) |
| LDW 误报率 | - | - | - | < 1次/100km |
| FCW lead dRel 误差 | < 15% (Carla) | < 18% (Carla) | < 15% (真实) | < 15% (真实) |
| FCW 漏报率 | - | - | - | < 5% |
| road_transform.z MAE | < 0.1m | < 0.15m | < 0.15m | - |
| 推理延迟 (NPU) | - | - | - | < 30ms |
| 模型大小 (INT8) | ~15MB (非部署) | < 5MB | < 5MB | < 5MB |

---

## 附录 A：源码引用索引

| 文件路径 | 关键行号 | 内容 |
|----------|----------|------|
| `selfdrive/controls/lib/ldw.py` | 6-8 | CAMERA_OFFSET, LDW_MIN_SPEED, LANE_DEPARTURE_THRESHOLD |
| `selfdrive/controls/lib/ldw.py` | 16-37 | LDW update 判断逻辑 |
| `selfdrive/modeld/fill_model_msg.py` | 143-149 | FCW 急刹预测滑动窗口 |
| `selfdrive/modeld/fill_model_msg.py` | 102-107 | 车道线输出解析 |
| `selfdrive/modeld/fill_model_msg.py` | 119-124 | Lead 输出解析 |
| `selfdrive/modeld/fill_model_msg.py` | 127-149 | Meta 输出解析与 FCW |
| `selfdrive/modeld/fill_model_msg.py` | 178-193 | Pose 输出填充 |
| `selfdrive/modeld/constants.py` | 6-26 | ModelConstants 定义 |
| `selfdrive/modeld/constants.py` | 29-32 | FCW 阈值定义 |
| `selfdrive/modeld/constants.py` | 74-87 | Meta 切片定义 |
| `selfdrive/modeld/parse_model_outputs.py` | 95-121 | 视觉网络输出解析 |
| `selfdrive/modeld/parse_model_outputs.py` | 123-128 | 策略网络输出解析 |
| `selfdrive/modeld/modeld.py` | 67-70 | 模型文件路径 |
| `selfdrive/modeld/modeld.py` | 198-304 | ModelState 类 |
| `selfdrive/controls/radard.py` | 26 | RADAR_TO_CAMERA = 1.52 |
| `selfdrive/controls/radard.py` | 141-156 | 纯视觉 lead 状态构建 |
| `selfdrive/controls/lib/longitudinal_planner.py` | 151 | FCW crash_cnt 判定 |
| `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` | 41 | CRASH_DISTANCE = 0.25 |
| `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` | 54 | FCW_IDXS = T_IDXS < 5.0 |
| `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` | 396-400 | MPC 碰撞检测逻辑 |
| `selfdrive/locationd/calibrationd.py` | 37-52 | 标定常量定义 |
| `selfdrive/locationd/calibrationd.py` | 52 | HEIGHT_INIT = 1.22 |
| `selfdrive/locationd/calibrationd.py` | 212-278 | handle_cam_odom 标定核心 |
| `selfdrive/locationd/calibrationd.py` | 259-260 | road_transform 高度更新 |
| `common/transformations/model.py` | 10-18 | MEDMODEL 参数 |
| `common/transformations/model.py` | 65-70 | get_warp_matrix 函数 |
| `common/transformations/camera.py` | 49-53 | 相机硬件参数（fcam focal=2648, ecam focal=567） |
| `common/transformations/camera.py` | 55-69 | DEVICE_CAMERAS 映射 |
| `common/transformations/camera.py` | 75-80 | 坐标系变换矩阵 |
| `tools/dashcam/run.py` | 70-71 | --camera-height 参数 |
| `tools/dashcam/run.py` | 87-88 | --road-only 单窄角相机模式 |
| `tools/dashcam/run.py` | 91-92 | --eval-lanes 评估 |
| `tools/dashcam/run.py` | 95-96 | --record 训练数据录制 |
| `tools/dashcam/run.py` | 99-101 | --speed-range 动态车速范围 |
| `tools/dashcam/run.py` | 102-103 | --record-only 纯录制模式 |
| `tools/dashcam/carla_world.py` | 15-19 | DashcamCarlaWorld 构造函数（含 speed_range, speed_interval） |
| `tools/dashcam/carla_world.py` | 277-284 | _update_speed() 动态速度控制 |
| `tools/dashcam/data_recorder.py` | — | DataRecorder：NPZ 训练数据保存 |
| `tools/dashcam/lane_ground_truth.py` | — | 车道线/路沿 GT 提取（Carla HD Map） |
| `tools/dashcam/lead_ground_truth.py` | — | 前车 GT 提取（Carla actor，[3,6,4] 格式） |
| `tools/dashcam/pose_ground_truth.py` | — | 视觉里程计 GT（帧间速度/角速度） |
| `tools/dashcam/view_npz.py` | — | NPZ 可视化工具（warp + GT 叠加） |
| `tools/dashcam/verify_pose.py` | — | pose 精度验证（v_ego 一致性、轨迹积分） |
| `tools/dashcam/warp_example.py` | — | rpyCalib warp 变换演示 |
| `selfdrive/modeld/models/driving_vision.onnx` | — | 视觉网络 ONNX (23M params, 45MB fp16) |
| `selfdrive/modeld/models/driving_policy.onnx` | — | 策略网络 ONNX (6.9M params, 14MB fp16) |
| `selfdrive/modeld/models/driving_vision_metadata.pkl` | — | 视觉网络输入输出元数据 |
| `selfdrive/modeld/models/driving_policy_metadata.pkl` | — | 策略网络输入输出元数据 |

## 附录 B：模型输出维度验证

基于 `constants.py` 中的常量定义验证输出维度：

```python
# 车道线
lane_lines_dim = NUM_LANE_LINES * IDX_N * LANE_LINES_WIDTH  # 4 * 33 * 2 = 264 (mean only)
lane_lines_prob_dim = NUM_LANE_LINES * 2  # 8 (raw logits, 取奇数索引得到4个概率)

# 路缘线
road_edges_dim = NUM_ROAD_EDGES * IDX_N * ROAD_EDGES_WIDTH  # 2 * 33 * 2 = 132 (mean only)

# 前车
lead_dim = LEAD_MHP_SELECTION * LEAD_TRAJ_LEN * LEAD_WIDTH  # 3 * 6 * 4 = 72 (mean only), MDN total = 72 * 2 = 144

# Meta (FCW 相关)
meta_fcw_dim = 5 + 5  # HARD_BRAKE_3(5) + HARD_BRAKE_5(5) = 10

# Pose
pose_dim = POSE_WIDTH  # 6

# Road Transform
road_transform_dim = POSE_WIDTH  # 6

# Desire Prediction
desire_pred_dim = DESIRE_PRED_LEN * DESIRE_PRED_WIDTH  # 4 * 8 = 32
```
