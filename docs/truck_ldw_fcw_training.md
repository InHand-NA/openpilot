# 卡车 LDW/FCW 视觉模型训练技术方案

## 目录

- [第1章：项目背景与需求分析](#第1章项目背景与需求分析)
- [第2章：openpilot LDW/FCW 实现分析](#第2章openpilot-ldwfcw-实现分析)
- [第3章：模型输出精简设计](#第3章模型输出精简设计)
- [第4章：网络架构设计](#第4章网络架构设计)
- [第5章：训练数据策略](#第5章训练数据策略)
- [第6章：损失函数设计](#第6章损失函数设计)
- [第7章：相机标定与高度估计](#第7章相机标定与高度估计)
- [第8章：模型部署方案](#第8章模型部署方案)
- [第9章：LDW/FCW 告警逻辑](#第9章ldwfcw-告警逻辑)
- [第10章：项目里程碑](#第10章项目里程碑)

---

## 第1章：项目背景与需求分析

### 1.1 openpilot supercombo 模型架构概述

openpilot 的驾驶模型采用**双网络架构**（`selfdrive/modeld/modeld.py`）：

1. **视觉网络（Vision Network）**：接收双目相机图像（窄角 + 广角），输出车道线、前车、视觉里程计等感知结果，以及一个 512 维的 `hidden_state` 特征向量。
2. **策略网络（Policy Network）**：接收视觉网络的 `hidden_state`、驾驶意图脉冲（`desire_pulse`）和交通规则信号，输出行驶轨迹规划（`plan`）和意图状态（`desire_state`）。

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
| 相机安装高度 | ~1.22m | 2.0-2.8m |
| 车身宽度 | ~1.8m | ~2.5m |
| 制动距离 (80km/h) | ~36m | ~70m+ |
| 载重变化 | 小 | 空载/满载高度差 10-20cm |
| FOV 特点 | 标准视角 | 高视角，更多地面可见 |
| 功能需求 | ACC + ALC（控制） | LDW + FCW（仅告警） |
| 部署硬件 | comma 3X (Qualcomm) | CPU/NPU 嵌入式平台 |

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

**卡车产品说明**：卡车方案无雷达，完全依赖此纯视觉路径。模型输出 `lead` 包含 6 个时间点（0/2/4/6/8/10s）的 4 维状态（x, y, v, a），以及 3 个 lead 目标的存在概率。

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
| `lead_prob` | BCE sigmoid | (3,) 或 (6,) | 前车存在概率 |
| `lead` | MDN (MHP) | MHP=2, selection=3, (6, 4) | 前车轨迹预测 |
| `hidden_state` | 直接传递 | (512,) | 传递给策略网络 |

**策略网络输出**（`parse_model_outputs.py:123-128`）：

| 输出名 | 解析方式 | 形状 | 说明 |
|--------|----------|------|------|
| `plan` | MDN (MHP) | MHP=5, selection=1, (33, 15) | 驾驶轨迹规划 |
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
| `lead` | MHP=2 时复杂 | ~200 (含 MHP 权重/std) | 视觉网络 | 前车 (x,y,v,a) × 6 时间点 |
| `lead_prob` | 3 或 6 | 6 | 视觉网络 | 前车存在概率 |
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
| `plan` | 5×33×15 MHP → ~5000D | 策略网络输出，控制规划用。仅告警不需要 |
| `desire_state` | 8D | 策略网络输出，车道变换状态机用 |
| `wide_from_device_euler` | 6D | 双目对齐，卡车系统可简化为单目标定 |
| `hidden_state` | 512D | 策略网络输入。不需策略网络时无意义 |
| `meta` (非 FCW 部分) | ~45D | ENGAGED、GAS/BRAKE_DISENGAGE、STEER_OVERRIDE 等，仅控制用 |

**精简后总输出维度估算**：

| 输出 | 维度（含 mean + std） |
|------|----------------------|
| lane_lines | 528 |
| lane_lines_prob | 8 |
| desire_pred | 32 |
| lead (简化为非 MHP) | 3×6×4×2 = 144 |
| lead_prob | 6 |
| meta (FCW only) | 10 |
| pose | 12 |
| road_transform | 12 |
| road_edges | 264 |
| **总计** | **~1016D** |

对比原始双网络总输出（视觉 + 策略合计数千维），**减少约 70-80%**。

### 3.6 策略网络整体移除

卡车产品不做自动驾驶控制（ACC + ALC），仅做 LDW/FCW 告警。因此：

- 策略网络**整体移除**，包括其全部输入（`features_buffer`、`desire_pulse`、`traffic_convention`）和输出（`plan`、`desire_state`）
- `hidden_state` 不再需要从视觉网络传递
- `InputQueues`（`modeld.py:123-196`）中为策略网络维护的时序缓冲也可移除

---

## 第4章：网络架构设计

### 4.1 架构选型

不再使用 openpilot 的双网络架构（视觉 + 策略），改用**单一视觉网络 + 多任务输出头**：

```
Input: 512×256 YUV (2 frames)    ← 纯图像输入，无需高度参数
       ↓
  ┌─────────────────────────────────────┐
  │ Backbone: EfficientNet-B0           │
  │         / MobileNetV3-Small         │
  └─────────────┬───────────────────────┘
                ↓
  ┌─────────────────────────────────────┐
  │ Temporal Fusion: ConvGRU /          │
  │   帧差拼接 + 1×1 Conv              │
  └─────────────┬───────────────────────┘
                ↓
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │Lane  │Lead  │Meta  │Pose  │Road  │Road  │Desire│
  │Lines │      │(FCW) │      │Trans │Edges │Pred  │
  └──────┘──────┘──────┘──────┘  ↑   └──────┘──────┘
                                 │
                          z 分量 = 相机高度估计
                          (供 calibrationd 在线更新)
```

### 4.2 Backbone 选项比较

| 模型 | 参数量 | FLOPS (224×224) | 特点 |
|------|--------|-----------------|------|
| MobileNetV3-Small | 2.9M | 56M | NPU 友好，量化后 ~0.7MB |
| EfficientNet-B0 | 5.3M | 390M | 精度更高，适中开销 |
| openpilot EfficientNet | ~14M | ~1.5G | 原始精度，不适合 NPU 部署 |

**建议**：起步使用 **EfficientNet-B0**，确保感知精度满足告警需求后，再通过知识蒸馏压缩到 MobileNetV3-Small 用于 NPU 部署。

输入分辨率保持 openpilot 的 `MEDMODEL_INPUT_SIZE = (512, 256)`（`common/transformations/model.py:10`），YUV 格式（NV12），双帧拼接后通道维度为 `2 × 6 = 12`（YUV 各 2 通道 × 2 帧）。

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

保留 openpilot 的 **2 帧输入设计**（`ModelConstants.N_FRAMES = 2`），使用轻量级时序融合：

**方案 A：帧差拼接（推荐起步方案）**
```
frame_t-1, frame_t → concat → [B, 12, H, W] → backbone
```
最简单高效，隐式编码运动信息。openpilot 原始模型也使用类似的多帧拼接方式。

**方案 B：ConvGRU（精度提升后考虑）**
```
frame_t-1 → backbone → features_t-1 ─┐
frame_t   → backbone → features_t   ──┼→ ConvGRU → fused_features
```
显式建模时序依赖，对高速场景的运动估计更准确。

**重要简化**：openpilot 原始设计中策略网络使用 `hidden_state`（512D）在帧间循环传递，这服务于控制决策的时序连贯性。卡车告警系统不需要此机制，每帧独立推理即可。

### 4.5 输出头设计

每个输出头独立解码，遵循 openpilot 的 MDN（Mixture Density Network）范式：

| 输出头 | 解码方式 | 原始输出维度 | 解析后维度 | 说明 |
|--------|----------|-------------|------------|------|
| lane_lines | MDN (mean + log_std) | 4×33×2×2 = 528 | 4×33×2 mean + 4 std | Laplace 分布 |
| lane_lines_prob | BCE sigmoid | 8 (raw logits) | 4 (取奇数索引) | 二分类概率 |
| lead | MDN (MHP=2, sel=3) | ~200 | 3×6×4 mean + std | 多假设混合 |
| lead_prob | BCE sigmoid | 6 | 3 概率 | 前车存在性 |
| meta_fcw | BCE sigmoid | 10 | 10 | HARD_BRAKE_3/5 各 5 时间点 |
| pose | MDN | 12 | 6 mean + 6 std | 视觉里程计 |
| road_transform | MDN | 12 | 6 mean + 6 std | 路面几何 |
| road_edges | MDN | 2×33×2×2 = 264 | 2×33×2 mean + 2 std | 路缘线 |
| desire_pred | CE softmax | 32 | 4×8 概率分布 | 意图预测 |

---

## 第5章：训练数据策略

### 5.1 三阶段数据构建

#### Phase 1：Carla 仿真数据（0-2 月）

利用已有的 `tools/dashcam/` 基础设施：

**已有能力**（`tools/dashcam/run.py`）：
- Carla 仿真器对接，支持多 Town 和天气条件
- VisionIPC 相机数据传输管线
- `--camera-height` 参数支持（`run.py:68-69`）
- `--eval-lanes` 车道线评估（`run.py:89-90`）
- `lane_ground_truth.py` 提供精确车道线 GT
- `lane_evaluator.py` 提供评估指标

**需扩展的能力**：
- 设置 `--camera-height 2.0/2.4/2.8` 模拟卡车视角
- 在多种高度之间随机采样，构建多高度混合数据集
- 提取 Carla 前车 actor 位置作为 lead GT（Carla 提供所有 actor 的 3D 位置和速度）
- 保存模型输入（YUV 帧 + 标定参数）和 GT 标签到训练数据格式

**数据量目标**：~50k 帧，覆盖：
- 高度：1.2m, 1.8m, 2.0m, 2.4m, 2.8m（5 个档位）
- Town：Town01-Town07（不同道路结构）
- 天气：晴天/阴天/雨天/雾天
- 时段：白天/黄昏/夜间

#### Phase 2：知识蒸馏（2-5 月）

使用 openpilot supercombo 模型作为 **teacher**：

```
                  Teacher (openpilot supercombo, frozen)
                         ↓ soft labels
Student (新网络) ← L_distill + L_gt(Carla) + L_temporal
```

**蒸馏策略**：
1. 在乘用车高度（1.22m）的 Carla 数据上，student 学习 teacher 的输出分布（soft labels），获得 teacher 对车道线和前车检测的"知识"
2. 在多高度 Carla 数据上，student 同时使用 Carla GT 进行监督学习
3. 两种损失联合优化，teacher 输出只在 1.22m 高度有效，GT 在所有高度有效

**为什么蒸馏有效**：
- Teacher 在百万级真实数据上训练，拥有强大的视觉特征提取能力
- 蒸馏 soft labels 传递了 teacher 对模糊/困难场景的"不确定性"信息
- Student 可以在更小的网络容量下接近 teacher 的性能

#### Phase 3：真实卡车数据微调（5-9 月）

- 在实际卡车上安装相机采集道路数据
- **自监督标注**：参考 openpilot 的 future-retrospective 方法：
  - 利用后续帧的视觉里程计（`pose` 输出）将未来观测投影回当前帧
  - 多帧时序累积提高远距离车道线精度
  - 前车检测可用成熟的 2D 检测器（如 YOLO）在图像上标注，再投影到 3D
- **半监督学习**：少量人工标注 + 大量自监督/伪标签数据

### 5.2 数据增强

| 增强类型 | 方法 | 目的 |
|----------|------|------|
| 几何 | 随机 pitch/yaw/roll 偏移 | 模拟安装偏差 |
| 高度 | 在 [1.0, 3.0]m 连续采样 | 覆盖高度范围 |
| 光照 | 亮度/对比度/色彩抖动 | 应对不同光照 |
| 遮挡 | 随机矩形遮挡 | 增强鲁棒性 |
| 模糊 | 运动模糊/高斯模糊 | 模拟真实退化 |

---

## 第6章：损失函数设计

### 6.1 总体损失

```
L_total = α·L_gt + β·L_distill + γ·L_temporal + δ·L_aux
```

其中各部分权重在训练阶段逐步调整。

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

### 6.3 蒸馏损失

```python
def distillation_loss(student_out, teacher_out, temperature=3.0):
    """KL 散度蒸馏损失"""
    # 对 MDN 输出：L2 距离（mean）+ KL（分布）
    L_mean = F.mse_loss(student_out['mean'], teacher_out['mean'])
    L_std = F.mse_loss(student_out['log_std'], teacher_out['log_std'])
    # 对概率输出：KL 散度
    L_prob = F.kl_div(
        F.log_softmax(student_out['logits'] / temperature, dim=-1),
        F.softmax(teacher_out['logits'] / temperature, dim=-1),
        reduction='batchmean') * (temperature ** 2)
    return L_mean + L_std + L_prob
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
| 模型大小（INT8） | < 3 MB | NPU 存储约束 |
| 推理延迟（NPU） | < 30 ms @ 20 FPS | 实时性要求 |
| 推理延迟（CPU） | < 100 ms | 降级运行 |
| 输入分辨率 | 512×256 YUV | 与 openpilot 一致 |
| 输出维度 | ~1016D | 精简后 |
| 精度要求（LDW） | 车道线 y MAE < 0.3m (0-60m) | 核心指标 |
| 精度要求（FCW） | lead dRel 误差 < 10% (0-100m) | 核心指标 |

### 8.3 运行时架构

```
Camera (YUV)
    ↓
Warp (calibration RPY) ← liveCalibration (rpyCalib)
    ↓
Model Input: [512×256 YUV × 2 frames]    ← 纯图像输入，无需高度参数
    ↓
┌───────────────────────────────────┐
│  Model Inference (NPU/CPU)        │
│  Backbone → Temporal → Heads      │
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

1. **算子兼容性**：确认目标 NPU 支持所有使用的算子（特别是 ConvGRU/GRU）
2. **内存布局**：NPU 通常偏好 NHWC 或特定对齐的 NCHW
3. **动态 shape**：避免动态 shape，所有输入输出形状固定
4. **后处理**：sigmoid/softmax 等激活函数尽量在 NPU 上完成

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

| 阶段 | 时间 | 目标 | 关键交付 |
|------|------|------|----------|
| **M1: 基础搭建** | 1-2月 | 数据管线 + 网络骨架 | Carla 多高度数据采集脚本；精简版网络定义（PyTorch）；训练框架搭建 |
| **M2: 仿真训练** | 2-4月 | Carla 数据训练达标 | LDW 车道线 MAE < 0.3m (0-60m)；FCW lead 检测 precision > 80% |
| **M3: 知识蒸馏** | 4-5月 | Teacher → Student 蒸馏 | 乘用车高度精度接近 teacher（>90%）；多高度精度保持 |
| **M4: 真实数据** | 5-7月 | 卡车实采数据微调 | LDW MAE < 0.5m（真实场景）；FCW lead 误差 < 15% |
| **M5: 部署优化** | 7-8月 | INT8 量化 + NPU 适配 | 模型 < 3MB；NPU 推理 < 30ms；量化精度损失 < 5% |
| **M6: 系统集成** | 8-9月 | 完整 LDW/FCW 系统 | 端到端系统测试；误报率 < 1次/100km；漏报率 < 5% |

### 验证标准

| 指标 | M2 目标 | M4 目标 | M6 目标 |
|------|---------|---------|---------|
| LDW 车道线 y MAE (0-60m) | < 0.3m (Carla) | < 0.5m (真实) | < 0.5m (真实) |
| LDW 误报率 | - | - | < 1次/100km |
| FCW lead dRel 误差 | < 15% (Carla) | < 15% (真实) | < 15% (真实) |
| FCW 漏报率 | - | - | < 5% |
| 推理延迟 (NPU) | - | - | < 30ms |
| 模型大小 (INT8) | - | - | < 3MB |

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
| `common/transformations/camera.py` | 49-53 | 相机硬件参数 |
| `common/transformations/camera.py` | 75-80 | 坐标系变换矩阵 |
| `tools/dashcam/run.py` | 68-69 | --camera-height 参数 |
| `tools/dashcam/run.py` | 89-90 | --eval-lanes 评估 |

## 附录 B：模型输出维度验证

基于 `constants.py` 中的常量定义验证输出维度：

```python
# 车道线
lane_lines_dim = NUM_LANE_LINES * IDX_N * LANE_LINES_WIDTH  # 4 * 33 * 2 = 264 (mean only)
lane_lines_prob_dim = NUM_LANE_LINES * 2  # 8 (raw logits, 取奇数索引得到4个概率)

# 路缘线
road_edges_dim = NUM_ROAD_EDGES * IDX_N * ROAD_EDGES_WIDTH  # 2 * 33 * 2 = 132 (mean only)

# 前车
lead_dim = LEAD_MHP_SELECTION * LEAD_TRAJ_LEN * LEAD_WIDTH  # 3 * 6 * 4 = 72 (mean only per hypothesis)

# Meta (FCW 相关)
meta_fcw_dim = 5 + 5  # HARD_BRAKE_3(5) + HARD_BRAKE_5(5) = 10

# Pose
pose_dim = POSE_WIDTH  # 6

# Road Transform
road_transform_dim = POSE_WIDTH  # 6

# Desire Prediction
desire_pred_dim = DESIRE_PRED_LEN * DESIRE_PRED_WIDTH  # 4 * 8 = 32
```
