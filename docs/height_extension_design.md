# 相机安装高度扩展技术方案（1m–3m）

> 目标：将当前基于 openpilot 预训练模型（~1.22m 标准高度）的 Dashcam 感知系统扩展到 1m–3m 安装高度，覆盖轿车 / SUV / 面包车 / 卡车等车型。

---

## 1. 背景与目标

### 1.1 当前局限性

openpilot 预训练 `driving_vision.onnx` 模型在 **~1.22m** 相机高度下训练（comma 3X 硬件规格）。

当相机安装高度发生改变时，输入图像的透视关系随之改变。核心问题在于 `get_warp_matrix` 使用的是**纯旋转单应矩阵**，不含高度平移分量：

```python
# common/transformations/model.py
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    calib_from_model = calib_from_medmodel          # 预计算，height=0 假设
    device_from_calib = rot_from_euler(device_from_calib_euler)   # 仅旋转
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model            # 无高度分量
    return warp_matrix
```

不同高度相机的地面特征经过 warp 后，在模型输入图像中位于不同位置，预训练模型无法正确解析。

### 1.2 目标

项目目前已经从openpilot 预训练 `driving_vision.onnx` 模型中提取并构建了./checkpoints/inadas_original.pt和./checkpoints/inadas_original.onnx，它们的推理精度与`driving_vision.onnx`基本一致。我们的目标是微调./checkpoints/inadas_original.pt，扩展到 1m–3m 安装高度，覆盖轿车 / SUV / 面包车 / 卡车等车型。

| 目标 | 描述 |
|------|------|
| 扩展高度范围 | 1.0m ~ 3.0m（轿车 1.0–1.3m，SUV 1.3–1.6m，面包车/卡车 2.0–3.0m） |
| 保留 LDW/FCW 功能 | 车道线检测（10–192m）和前车检测（5–30m）精度不显著退化 |
| 在线兼容 | 高度信息可从 calibrationd 在线估计，无需手动配置 |
| 最小侵入性 | 利用预训练权重，仅扩展模型轻量化头部 |

### 1.3 关键评价指标

- 车道线横向误差 MAE（按距离分段）
- 前车检测 mAP（BEV，IOU > 0.5）
- 各高度档位独立评价 + 跨高度泛化测试

---

## 2. 技术分析

### 2.1 warp 变换的高度误差分析

#### 理论推导

对于相机高度差 ΔH，在距离 X 处观察地面点，高度引起的仰角视差为：

```
δθ ≈ arctan(ΔH / X)
```

对应图像平面上的像素位移（焦距 f = 910px）：

```
δy_px ≈ f * ΔH / X
```

#### 量化分析

以标准高度 H₀ = 1.22m，卡车高度 H₁ = 3.0m（ΔH = 1.78m）为例：

| 距离 X | 仰角视差 δθ | 像素位移 δy | 对 LDW/FCW 影响 |
|--------|------------|------------|----------------|
| 5m     | 19.6°      | ~311px     | **严重**（近场前车） |
| 10m    | 10.2°      | ~162px     | **显著**（近场车道线） |
| 20m    | 5.1°       | ~81px      | **中等** |
| 30m    | 3.4°       | ~54px      | **中等** |
| 50m    | 2.0°       | ~32px      | **较小** |
| 80m    | 1.3°       | ~20px      | **较小** |
| 100m   | 1.0°       | ~16px      | **可接受** |
| 192m   | 0.5°       | ~8px       | **可忽略** |

**结论**：
- **近场（X < 20m）**：误差最大，FCW 受影响最严重
- **中场（20m–80m）**：LDW 主要工作区间，误差中等，需要修正
- **远场（X > 80m）**：误差趋向可接受

#### 对各功能的影响优先级

```
FCW（5–30m） > LDW近场（10–30m） > LDW中场（30–80m） > LDW远场（>80m）
```

### 2.2 模型输出的高度无关性

模型输出采用**设备帧坐标系**（device frame），原则上高度无关：

```
lane_lines:    (4, 33, 2) = [y_lateral, z_height] in device frame
road_edges:    (2, 33, 2) = [y_lateral, z_height] in device frame
lead:          (3, 6, 4)  = [x_fwd, y_lat, v_rel, a_rel] in device frame
pose:          (6,)        = [translation(3) + rotation(3)]
road_transform:(6,)        = 路面几何相对设备帧变换
```

这意味着：**相同场景的不同安装高度数据，可以共用同一套 GT 标注约定（设备帧坐标系）**，为多高度联合训练奠定基础。

### 2.3 calibrationd 的在线高度估计

`calibrationd.py` 已实现从 `road_transform_trans[2]` 提取高度：

```python
# calibrationd.py
HEIGHT_INIT = np.array([1.22])  # 默认初值（米）

# 在线更新：从 modeld 输出的 road_transform 平移 z 分量提取
new_height = np.array([road_transform_trans[2]])
liveCalibration.height = self.height.tolist()  # 发布到消息总线
```

这意味着：**系统在线运行时可自动估计相机高度**（前提：modeld 输出准确的 `road_transform`），无需用户手动配置高度参数。

### 2.4 现有数据格式的高度支持

`dual_data_recorder.py` 已在每帧数据中存储高度字段：

```python
# dual_data_recorder.py — NPZ 输出格式
save_dict['camera_height'] = np.float32(self.camera_height)  # 已有字段
```

数据集和训练代码扩展时只需在加载侧读取该字段即可。

---

## 2.5 隐式高度适应 vs 显式高度注入：技术方案深度对比

### 问题提出

一个重要的设计替代方案：现有模型已通过 `road_transform_trans[2]` 间接输出相机安装高度。若训练数据覆盖多个高度档位（H1–H6），模型能否**隐式地**学习到高度信息，用**一个不含高度输入的模型**适应多种高度？这样可以省去 HeightConditionedHead，简化部署。

本节对"隐式高度适应"与"显式高度注入"两种方案进行系统性分析。

---

### 2.5.1 隐式方案（多高度联合训练，无显式高度输入）

**核心假设**：不同安装高度对应不同俯仰角 → 不同 warp 矩阵 → 不同的视觉输入分布。模型通过看到多高度数据，自然学会从图像透视关系推断高度并产生正确输出。

#### 支持隐式方案的论点

**1. 视觉输入本身携带高度信息（pitch 编码高度）**

在采集方案中，高度与俯仰角绑定（`pitch = arctan(H/15)`），不同高度产生**系统性不同**的 warped 图像：

| 高度 H | 俯仰角 | 近场地面特征占比 | 视觉特点 |
|--------|--------|----------------|---------|
| 1.0m | −3.8° | 小 | 接近人眼视角 |
| 1.5m | −5.7° | 中 | — |
| 3.0m | −11.3° | 大 | 近似鸟瞰感 |

不同高度的图像在视觉上可区分，原则上模型可以从透视关系**隐式推断**当前高度。

**2. backbone 内部表征已隐式编码高度**

`road_transform_trans[2]` 输出相机高度这一事实表明：backbone 23M 参数的内部特征向量中已经包含足够的高度信息，只是没有被显式利用来调制其他输出头。这说明"让模型隐式感知高度"在原理上并不困难——它已经在做了。

**3. 设备帧坐标系使关键输出高度无关**

所有模型输出均在设备帧（device frame）下，对 LDW/FCW 最关键的分量分析：

| 输出分量 | 是否随高度变化 | 分析 |
|---------|--------------|------|
| `lane_lines[:, :, 0]`（y_lat 横向） | **否** | 车道线横向位置与相机高度无关 |
| `lane_lines[:, :, 1]`（z_height） | 是 | z_device ≈ −camera_height（线性相关） |
| `lead.x_fwd`（前车纵向距离） | 极小 | 车身高度约 1.5m，在 5–30m 距离下高度效应 < 3° |
| `lead.y_lat`（前车横向） | **否** | 与相机高度无关 |
| `pose`（运动状态） | **否** | 纯运动学，与高度无关 |

**结论：LDW（y_lat）和 FCW（x_fwd, y_lat）最关键的输出分量本身就是高度无关的。** 隐式方案在这些维度上不存在原理性障碍。

**4. 实现最简单，部署无额外依赖**

隐式方案无需修改模型架构，无需在线高度估计，模型完全自包含。

---

#### 反对隐式方案的论点（关键限制）

**⚠️ 限制1：calibrationd 的 PITCH_LIMITS 硬约束（最重要）**

`calibrationd.py` 中存在俯仰角上限硬约束：

```python
PITCH_LIMITS = np.array([-0.09074112085129739, 0.17])  # 弧度，约 [-5.2°, +9.74°]
```

而不同高度对应的俯仰角需求：

| 高度 H | 理论俯仰角 | 弧度值 | 是否超出限制 |
|--------|----------|--------|------------|
| 1.0m | 3.8° | 0.0663 | ✅ 正常 |
| 2.0m | 7.6° | 0.1328 | ✅ 正常 |
| 2.5m | 9.46° | 0.1651 | ⚠️ 临界（0.17 限制） |
| **3.0m** | **11.31°** | **0.1974** | **❌ 超出限制！** |

**后果**：对于 H ≥ 3.0m 的相机，calibrationd 会将俯仰角截断为 ~9.74°，导致 warp 矩阵计算错误——输入图像的透视关系与实际高度不符，完全破坏了隐式方案赖以成立的"pitch 编码高度"假设。

这是**部署层面的系统性缺陷**，不是边缘情况。

**反驳观点**： 这个反对观点认为在所有安装高度下，视觉都应该关注15m地面。 但根据人类驾驶经验，重型卡车视角更高，制动距离也更远，应该关注更远的距离。

**限制2：pitch–height 解耦，部署时对应关系不保证**

训练时：`pitch = arctan(H/15)`（理想化假设，相机注视 15m 前方地面）。

部署时：pitch 由 calibrationd 从 road_transform 的输出动态估计，**不保证遵循此公式**。实际安装时：
- 卡车驾驶员可能将相机固定在某个安装架上，pitch 角由安装架决定
- 同一高度（如 2.5m）的不同安装方式可能产生完全不同的 pitch 角
- calibrationd 收敛需要时间，初始阶段 pitch 估计不准确

一旦 pitch ≠ arctan(H/15)，训练时建立的视觉-高度对应关系就不再成立，隐式方案面临**分布外（OOD）问题**。

**应对方案**：`pitch = arctan(H/15)`只是用来计算Pitch边界值的方式，不是实际训练配置。应该采集多种不同pitch–height 组合的数据微调模型。

**限制3：calibrationd 高度估计的"鸡和蛋"问题**

即使只是想用 calibrationd 的高度估计（而非 pitch 编码）来辅助隐式方案：

```
calibrationd 使用 road_transform_trans[2] 来估计高度
但 road_transform 的准确性依赖于模型在当前高度下已正确运行
→ 模型初始在 1.22m 下训练，在 3.0m 下 road_transform 输出不准确
→ calibrationd 高度估计不准确
→ 循环依赖
```

这意味着：即便在隐式方案中想利用 `calibrationd.height` 作为外部高度提示，也存在冷启动问题。

**应对方案**：部署时使用人为输入的高度数据作为`calibrationd.height`初值，解决冷启动问题。

**限制4：中间高度插值能力差**

隐式方案以 6 个离散档位训练，模型在档位之间（如 H=2.3m）的行为不保证单调、平滑或正确。显式方案通过连续高度嵌入（HeightEmbedding）天然提供平滑插值。

**应对方案**：如有需要，可以设计更多更细的高度档位。

**限制5：训练数据效率低**

隐式方案要求模型自行从透视差异中推断高度，增加学习难度，需要更多数据才能达到相同精度。

**应对方案**： 看评估效果再做决定。

---

### 2.5.2 显式方案（HeightConditionedHead）

**核心思路**：将相机高度作为额外输入，通过 HeightEmbedding + MLP 在 backbone 特征层面直接调制输出，消除高度歧义。

#### 显式方案的关键优势

**1. 直接消除高度歧义**

高度直接作为条件输入，模型不需要从图像中"猜"自己在哪个高度。即使 warp 矩阵因 pitch 估计不准而存在偏差，高度输入仍能提供正确的"期望输出"约束。

**2. 连续高度空间，平滑插值**

HeightEmbedding 将标量高度映射为连续嵌入向量（正弦编码），模型在训练高度档位之间的插值行为在理论上是平滑的。

**3. 数据效率高**

高度直接提供，模型可专注于学习"给定高度下的视觉到输出映射"，而非同时学习高度推断和输出校正两个任务。

**4. 对 warp 误差具有补偿能力**

当 pitch 估计偏差导致 warp 矩阵有误时，显式高度输入可为输出头提供补偿信号（实际上在学习"高度偏差 + warp 误差 → 输出修正"的联合映射）。

#### 显式方案的关键限制

**1. 依赖 calibrationd 准确输出高度**

calibrationd 需要经过一段时间（≥ INPUTS_NEEDED × BLOCK_SIZE = 500 帧 ≈ 25 秒）才能收敛到准确的高度估计。冷启动阶段，模型收到的高度输入不准确，性能下降。

**应对方案**：部署时使用人为输入的高度数据作为`calibrationd.height`初值，解决冷启动问题。

**2. 参数量轻量化的代价**

HeightConditionedHead 只有 ~250K 参数，backbone（23M）保持冻结。这意味着：
- backbone 的特征空间仍然是在标准高度（1.22m）下训练的
- 对高度差异极大（H=3.0m）的场景，轻量级头部的调制能力可能不足
- 可能需要解冻部分 backbone 进行全量微调才能达到最优性能

---

### 2.5.3 方案对比矩阵

| 评估维度 | 隐式方案（多高度联合训练） | 显式方案（HeightConditionedHead） |
|---------|--------------------------|----------------------------------|
| **架构改动** | 零（直接使用现有模型微调） | 新增 HeightEmbedding + MLP (~250K参数) |
| **部署依赖** | 无（模型自包含） | 需 calibrationd 提供高度（25s 冷启动） |
| **H ≤ 2.5m 可行性** | ✅ 高（pitch 在限制范围内） | ✅ 高 |
| **H = 3.0m 可行性** | ✅ 高（恒定 5° 策略下 pitch=5°，在 PITCH_LIMITS 范围内） | ✅ 高 |
| **pitch-height 解耦鲁棒性** | 训练数据的配置组合多样化 | ✅ 好（高度独立输入，可部分补偿 pitch 误差） |
| **中间高度插值** | ❌ 差（离散档位训练，无连续性保证） | ✅ 好（连续高度嵌入，平滑插值） |
| **LDW 横向精度** | ✅ 高（y_lat 本身高度无关） | ✅ 高 |
| **FCW 精度** | ✅ 高（x_fwd 高度无关） | ✅ 高 |
| **训练数据效率** | 低（需从视觉隐式推断高度） | 高（高度直接作为条件） |
| **实现复杂度** | 低 | 中 |
| **部署风险** | 中（H≥3.0m 有系统性失效风险） | 低-中（冷启动期间高度不准） |

---

### 2.5.4 关键结论

**隐式方案的可行性评估**：

- 对 **H ≤ 2.5m** 场景：隐式方案在原理上可行。pitch 在 calibrationd 的允许范围内，视觉差异可被模型学习。对最关键的 LDW（y_lat）和 FCW（x_fwd）输出，高度无关性进一步降低了学习难度。**值得作为第一阶段基线验证**。

- 对 **H = 3.0m** 场景：采用恒定 5° 俯仰角策略（§2.9）后，pitch = 5° = 0.087 rad，完全在 PITCH_LIMITS 范围内，PITCH_LIMITS 问题不复存在。隐式方案能否应对 H=3.0m 的视觉差异，取决于训练数据的覆盖和模型表达能力。

**显式方案的必要性**：

- 对需要**精确插值**（H=2.3m 等中间值）、**OOD 高度泛化**或**极端高度（H=3.0m）**的场景，显式方案不可替代。
- 显式方案的代价（250K 参数 + calibrationd 高度估计）完全可接受，且 calibrationd 已有成熟的在线高度估计能力。

---

### 2.5.5 推荐策略：隐式优先，显式增强

**分阶段实施**：

```
第一阶段（隐式基线）：
  → 多高度联合训练（H1–H6）
  → 不添加高度输入，直接微调 PretrainedVisionModel
  → 量化各高度的精度，建立基线

  判断标准：
  - H2 (1.3m) 车道线近场 MAE < 0.10m  → 基线参考
  - H4 (2.0m) 车道线近场 MAE < 0.20m  → 可接受
  - H6 (3.0m) 车道线近场 MAE < 0.30m  → 最低要求
  - H5 (2.5m, 泛化测试) MAE < H4 和 H6 的线性插值 → 插值能力验证

若第一阶段满足标准 → 可直接部署隐式模型，节省显式方案的工程复杂度
若第一阶段不满足（预期 H5/H6 场景） → 进入第二阶段

第二阶段（显式增强）：
  → 在第一阶段权重基础上添加 HeightConditionedHead
  → 冻结微调后的 backbone，只训练 head
  → 或端对端全量微调（lr=1e-5）
```

这种策略同时获得了**隐式方案的简洁性**（如果精度够用）和**显式方案的精度上界**（如果需要）。

---

## 2.6 隐式方案的 backbone 微调必要性与数据量分析

### 问题提出

隐式方案选择直接微调 `PretrainedVisionModel`（openpilot ONNX，23M参数）。两个核心问题：
1. **必须微调 backbone 权重吗？** 还是冻结 backbone、只微调输出头就够？
2. **需要多少训练数据？**

---

### 2.6.1 backbone 是否必须微调：分布偏移分析

#### 高度变化造成的视觉分布偏移

openpilot backbone 在 **H₀ = 1.22m** 标准高度、数百万英里真实驾驶数据下训练。对于不同高度的输入，Warp 矩阵校正了俯仰角（图像旋转），但**不校正高度引起的近场地面缩放**。

Warp 后的输入图像（128×256）在不同高度下的实际分布变化：

| 高度 H | ΔH | X=10m 像素偏移 | 分布偏移程度 | 视觉特征变化 |
|--------|-----|-------------|-----------|-----------|
| 1.22m | 0 | 0px | **无（基准）** | openpilot 训练基准高度 |
| 1.3m | +0.08m | 7px | **几乎可忽略** | 肉眼几乎不可见 |
| 1.5m | +0.28m | 25px | **小** | 近场特征轻微位移 |
| 2.0m | +0.78m | 71px | **中** | 近场车道线上移约 1/4 图像高度 |
| 2.5m | +1.28m | 116px | **大** | 近场特征大范围位移，透视感明显不同 |
| 3.0m | +1.78m | 162px | **极大** | 近场特征上移超过一半图像高度，呈近鸟瞰感 |

#### 冻结 backbone 时的特征质量退化

Backbone 内部卷积层的特征提取具有**空间位置敏感性**：

```
输入图像 (128×256)
  ↓ Stem (stride=4): (32, 64)
  ↓ Stage 0-3 (各层渐进 downsample): (4, 8)
  ↓ GAP → FC(2048) → 977维输出
```

当 H=3.0m 时，近场车道线标记（X=10m 处）从标准位置上移 162px，已超过整个图像高度（128px）的 **1.26 倍**。这意味着：

- **标准高度（1.22m）下**：backbone 的卷积核已学会在特定空间位置检测车道线（中下部区域）
- **H=3.0m 下**：车道线位于完全不同的空间位置（中部区域），backbone 的空间感受野已无法在正确位置激活
- **结果**：backbone 在 OOD（out-of-distribution）输入下产生降质特征，输出头无法从错误的特征向量中恢复正确预测

这与"backbone 已经输出 road_transform 高度"并不矛盾——该能力是在 1.22m 数据上学到的高度估计，而不是对任意高度的鲁棒特征提取。

#### 各高度档位的 backbone 微调必要性判断

| 高度档位 | 10m像素偏移 | backbone 是否必须微调 | 理由 |
|---------|-----------|-------------------|------|
| H1 (1.22m) | 0px | **否（基准）** | openpilot 训练高度，零分布偏移 |
| H2 (1.3m) | 7px | **否** | 几乎在训练分布内 |
| H3 (1.5m) | 25px | **否（建议微调头部）** | 小偏移，冻结backbone可接受 |
| H4 (2.0m) | 71px | **推荐微调（部分层）** | 中等偏移，冻结时近场特征质量下降 |
| H5 (2.5m) | 116px | **必须微调** | 大偏移，冻结backbone特征严重退化 |
| H6 (3.0m) | 162px | **必须全量微调** | 超大偏移，空间特征完全失配 |

**关键结论**：对于目标覆盖 H1–H6 的隐式方案，**backbone 必须微调**（至少部分层）。仅微调输出头无法处理 H ≥ 2.0m 的场景。

---

### 2.6.2 哪些层需要微调：层级分析

#### Backbone 层级结构（openpilot 23M参数，推测架构）

```
底层（low-level）：Stem + Stage0 / early conv
  → 检测边缘、纹理、颜色分布
  → 相对高度无关（车道线的颜色/对比度不随高度变化）
  → 建议：可以冻结或用极小 lr

中层（mid-level）：Stage1-2
  → 建立空间几何关系、车道线段检测
  → 部分高度相关（空间尺度变化）
  → 建议：中等 lr（1e-5 级别）

高层（high-level）：Stage3 + FC
  → 整合全局视觉特征，建立语义理解
  → 显著高度相关（"前方有车道线"→"这是X米处的车道线"）
  → 建议：较大 lr（3e-5 级别），必须微调

输出头（output heads）：
  → 直接映射特征到预测输出
  → 必须微调，lr 可较大（1e-4 级别）
```

#### 推荐的分层微调策略

```python
# 分层学习率（推荐配置）
optimizer = torch.optim.AdamW([
    {'params': backbone.stem.parameters(),    'lr': 0},       # 冻结
    {'params': backbone.stages[:2].parameters(), 'lr': 1e-5}, # 低层，极小lr
    {'params': backbone.stages[2:].parameters(), 'lr': 3e-5}, # 高层，小lr
    {'params': backbone.fc.parameters(),      'lr': 5e-5},    # FC层
    {'params': output_heads.parameters(),     'lr': 1e-4},    # 输出头
], weight_decay=1e-4)
```

---

### 2.6.3 训练数据量估算

#### 估算框架

数据量需求由以下因素决定：
1. **分布偏移程度**（越大需要越多数据）
2. **预训练模型质量**（越强需要越少数据）
3. **任务有效复杂度**（LDW y_lat 是高度无关的，FCW x_fwd 基本高度无关）
4. **标注质量**（Carla GT 比 modeld 伪标签需要更少数据）

#### 关键洞察：有效学习任务比想象简单

对于模型最关键的输出（LDW 横向位置 y_lat，FCW 纵向距离 x_fwd），它们在设备帧下**本身就是高度无关的**。backbone 微调的主要目标是：

> **让模型在新高度的 warped 图像中，找到与标准高度对应的视觉特征位置**

这本质上是一个"空间重新校准"任务，而非学习全新的语义概念。这使得数据需求比完整领域适应（domain adaptation）低一个数量级。

#### 各高度档位数据需求估算

| 高度档位 | 分布偏移 | 微调策略 | 最少有效帧数 | 推荐帧数 | 理由 |
|---------|---------|---------|-----------|---------|------|
| H1 (1.22m) | 无（基准） | 仅输出头 | **200** | 1000 | openpilot 训练高度，作为微调锚点 |
| H2 (1.3m) | 几乎无 | 仅输出头 | **500** | 1000 | 接近原始训练高度，可做锚点 |
| H3 (1.5m) | 小 | 头 + 高层 | **1000** | 3000 | 需要头部适应 |
| H4 (2.0m) | 中 | 头 + 中高层 | **3000** | 5000 | 71px 偏移，需要中层适应 |
| H5 (2.5m) | 大 | 全量微调 | **5000** | 8000 | 116px 偏移，需要全层适应 |
| H6 (3.0m) | 极大 | 全量微调 | **8000** | 10000 | 162px 偏移，需充分数据防止忘记 |
| **合计** | | | **~18000** | **~28000** |  |

#### 场景多样性修正

上述数据量基于 **≥3 种地图、混合天气/时间** 的条件。若场景单一：
- 仅 1 种地图：需要 **2–3×** 上述数据量（否则过拟合单一场景）
- 仅晴天白天：需要 **1.5×** 数据量

#### 时序考虑：帧对而非独立帧

模型输入是连续帧对（current + prev，间隔 4 帧）。有效独立样本数约为总帧数的 **1/4**。若视频以 20FPS 录制：
- 30 分钟 Carla 录像 ≈ 36,000 帧 ≈ **9,000 个有效独立样本**
- 每高度 5000 帧 = 约 3–4 分钟 Carla 连续录像

这意味着：**每个高度档位的数据采集量（5000帧）对应约 4 分钟的 Carla 仿真**，采集成本很低。

---

### 2.6.4 灾难性遗忘（Catastrophic Forgetting）风险

全量微调 23M 参数的最大风险：在新高度数据上过训练，导致在标准高度（H=1.22m）性能退化。

**缓解策略**：

1. **锚定高度**：在所有训练批次中始终包含 H1（1.22m）和H2（1.3m）数据，用于防止标准高度性能退化。H1+H2 数据量占比建议不低于 **20%**（采用加权采样）。

2. **分层学习率**：底层学习率远小于输出层（见 §2.6.2），防止低层特征被破坏。

3. **L2 正则化**：对 backbone 参数施加较大 weight_decay（1e-4），约束权重偏离预训练值。

4. **Early stopping 基于 H1+H2 验证损失**：若 H1+H2验证损失上升超过 5%，立即停止并 rollback。

5. **渐进式解冻**：
   ```
   Epoch 1-10:   仅解冻输出头，lr=1e-4
   Epoch 11-20:  解冻 Stage3 + FC，lr=3e-5（头继续 1e-4）
   Epoch 21-40:  解冻 Stage2，lr=1e-5
   Epoch 41+:    可选：全量解冻，lr=5e-6
   ```

---

### 2.6.5 数据量与精度关系的预期曲线

基于迁移学习经验，预期各高度的精度随数据量变化规律：

```
H1/H2（小偏移）：
  1000 帧  → 精度接近标准高度（~95%）
  1000 帧 → 收敛（无需更多）

H4（中等偏移）：
  1000 帧 → 明显改善（~70% 精度）
  3000 帧 → 接近收敛（~85% 精度）
  5000 帧 → 收敛（~90% 精度）

H6（大偏移）：
  2000 帧 → 基本可用（~60% 精度）
  5000 帧 → 中等（~75% 精度）
  10000 帧 → 接近收敛（~85% 精度）
```

> 注："X% 精度"指与 H2 标准高度基线相比的相对 LDW MAE 倒数。

#### 关键数据量拐点

- **< 1000 帧（单高度）**：仅头部适应阶段，对 H ≥ 2.0m 不足
- **3000–5000 帧（单高度）**：推荐最小目标，覆盖 H1–H4
- **8000–10000 帧（单高度）**：高质量训练，覆盖 H5–H6
- **> 10000 帧（单高度）**：收益递减，优先增加场景多样性

---

### 2.6.6 总结

| 问题 | 结论 |
|------|------|
| **是否需要微调 backbone？** | **是**，对 H ≥ 2.0m 必须微调至少高层；H ≤ 1.5m 可仅微调输出头 |
| **微调哪些层？** | 渐进式解冻：头部 → 高层 → 中层；底层 Stem 可保持冻结 |
| **需要多少数据？** | H1–H3：各 1000–3000 帧；H4：5000 帧；H5–H6：8000–10000 帧；**总计约 25000–30000 帧** |
| **设计文档 5000帧/高度 目标是否合适？** | **基本合适**：覆盖 H1–H4，H5–H6 略显不足，建议 H5/H6 增至 8000 帧 |
| **最大风险** | 全量微调时的灾难性遗忘 → 用锚定高度（H1）+ 渐进解冻 + Early stopping 缓解 |
| **单高度数据采集成本** | 5000 帧 ≈ 4 分钟 Carla 仿真，采集成本极低 |

---

## 2.7 PITCH_LIMITS 设计值溯源

### 问题提出

`calibrationd.py` 中硬编码了俯仰角限制：

```python
PITCH_LIMITS = np.array([-0.09074112085129739, 0.17])  # 弧度
```

注释说明这些值"为了让窄摄像头容纳模型帧（to accommodate the model frame in the narrow cam）"。本节分析这些值的几何来源。

---

### 2.7.1 PITCH_LIMITS 的几何推导

#### 涉及参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `camera_fl` | 2648.0 px | tici 窄摄像头焦距（AR0231/OX03C10） |
| `camera_CY` | 604.0 px | 窄摄像头主点纵坐标（1208/2） |
| `camera_height` | 1208 px | 窄摄像头图像高度 |
| `medmodel_fl` | 910.0 px | medmodel 坐标系焦距 |
| `MEDMODEL_CY` | 47.6 px | medmodel 主点纵坐标（靠近图像顶部） |
| `medmodel_height` | 256 px（显示为128行YUV后） | 模型输入图像高度 |

**关键比例**：`camera_fl / medmodel_fl = 2648.0 / 910.0 ≈ 2.910`

#### Warp 矩阵的线性近似

在俯仰角 θ（正值=向下，负值=向上）时，模型输入行 r 近似对应相机图像行：

```
camera_row(r) ≈ camera_CY + (r - MEDMODEL_CY) × (camera_fl/medmodel_fl) − θ × camera_fl
```

在 θ = 0 时（水平）：

```
camera_row(0)   = 604 − 47.6 × 2.910 = 604 − 138.5 = 465.5    # 模型顶行（远场）
camera_row(127) = 604 + (127−47.6) × 2.910 = 604 + 231.1 = 835.1  # 模型中行
camera_row(255) = 604 + (255−47.6) × 2.910 = 604 + 603.5 ≈ 1207.5 # 模型底行（近场），恰好在相机边缘！
```

这一精确对齐不是巧合——**medmodel 的 MEDMODEL_CY = 47.6 是精心设计的，使得在标准安装角度（θ≈4.65°）下，模型底行近场和顶行远场分别对应相机图像的下边缘和安全区域**。

#### 上限 PITCH_LIMITS[1] = 0.17 rad 的几何推导

**约束**：模型输入**顶行（r=0，对应远场）**必须能从相机图像内采样（camera_row ≥ 0）：

```
camera_row(0) ≥ 0
465.5 − θ × 2648 ≥ 0
θ ≤ 465.5 / 2648 = 0.17579 rad ≈ 10.07°
```

**几何上限 = 0.1758 rad（10.07°）**，openpilot 取 **0.17 rad（9.74°）**，预留了 **0.34° 安全余量**。

**物理含义**：当相机向下俯仰角超过 10.07° 时，模型的远场顶行需要从相机图像上边缘**之外**的区域采样——这在物理上不可能，warp 会用零值填充，导致模型远场感知失效。

#### 下限 PITCH_LIMITS[0] = −0.09074 rad 的设计依据

从几何约束推导此值的具体来源较复杂，但其数量级有清晰的物理意义：

在 θ = −0.09074 rad（向上仰角 5.2°）时，模型底行（近场，r=255）已映射到相机图像外：
```
camera_row(255) = 1207.5 + 0.09074 × 2648 ≈ 1448  > 1207（相机高度）
```

因此该下限**不是**"所有像素均在相机内"的约束（该约束几乎要求 θ ≥ 0），而是一个**经验性上界**——在真实车辆中，行车记录仪向上仰角超过 5.2° 属于异常安装，不在校准系统的考虑范围内。此值可理解为：**保留足够的远场可视性**（模型顶行在相机内），同时允许轻微向上安装偏差。

#### 数值验证汇总

```
θ = 0.00 rad (0°):      camera_row(0) = 465.5  ✅ 模型顶行在相机内
θ = 0.17 rad (9.74°):   camera_row(0) ≈  15.6  ✅ 刚好在相机内（几何上限）
θ = 0.1758 rad (10.07°): camera_row(0) ≈   0    ⚠️ 临界
θ = 0.1974 rad (11.31°): camera_row(0) ≈ -57.2  ❌ H=3.0m 情况，约20行采样失效
```

---

## 2.9 卡车场景的注视距离分析

### 问题提出

当前设计沿用乘用车的标准：以 **15m 前方地面**为注视中心来计算俯仰角（`pitch = arctan(H/15)`）。但基于人类驾驶经验，重卡 / 长途卡车驾驶员的自然注视距离远大于 15m。本节分析是否应为卡车场景采用更大的注视距离。

---

### 2.9.1 人类驾驶经验分析

重卡驾驶员的注视行为与乘用车有本质区别：

| 车型 | 安装高度 | 典型速度 | 制动距离 | 驾驶员自然注视距离 |
|------|---------|---------|---------|----------------|
| 小轿车 | 1.22m | 100 km/h | ~42m | 40–60m |
| SUV | 1.5m | 90 km/h | ~38m | 40–60m |
| 中型货车 | 2.0m | 80 km/h | ~50m | 50–80m |
| 大型货车 | 2.5m | 80 km/h | ~80m | 60–100m |
| **重卡/长途** | **3.0m** | **80 km/h** | **~120m** | **80–150m** |

**关键驾驶经验**：
- 商业卡车驾驶培训要求注视**前方 12–15 秒**的路况，80 km/h 对应 **267m**
- 实际操作中，驾驶员持续扫视 80–150m 前方以应对制动需求
- 15m 的注视距离对于卡车驾驶员来说约等于看着自己的车头，在实际驾驶中不具意义

---

### 2.9.2 注视距离对相机俯仰角的影响（H=3.0m）

H=3.0m 时，不同注视距离对应的俯仰角与几何参数：

| 注视距离 | 俯仰角 | PITCH_LIMITS | 模型顶行在相机内的位置 | 最近可见 X |
|---------|--------|-------------|---------------------|---------|
| 15m（当前） | 11.31° | ❌ 超出（需扩展） | −57px（相机外） | 6.7m |
| 20m | 8.53° | ✅ 在限制内 | +71px | 7.7m |
| 25m | 6.84° | ✅ 在限制内 | +149px | 8.4m |
| 30m | 5.71° | ✅ 在限制内 | +202px | 8.9m |
| **34m（推荐）** | **5.00°** | **✅ 在限制内** | **+232px** | **9.3m** |
| 50m | 3.43° | ✅ 在限制内 | +307px | 10.3m |

将注视距离从 15m 改为 ≥ 20m，俯仰角即降至 8.53°，完全在现有 PITCH_LIMITS（9.74°）内，无需任何 PITCH_LIMITS 修改。

---

### 2.9.3 恒定俯仰角策略（推荐方案）

**核心思想**：不同安装高度下，保持相机俯仰角近似不变（≈5°），注视距离随高度等比例增大。

```python
TARGET_PITCH_DEG = 5.0  # 与标准乘用车基本一致（1.22m 时 4.65°）
look_at_distance = H / math.tan(math.radians(TARGET_PITCH_DEG))
```

各高度档位的计算结果：

| 高度 H | 注视距离 look_at | 俯仰角 pitch | 模型顶行位置 | 最近可见 X | PITCH_LIMITS |
|--------|---------------|------------|-----------|---------|-------------|
| 1.22m | 13.9m ≈ **14m** | 5.00° | +234px | 3.8m | ✅ |
| 1.5m | 17.1m | 5.00° | +234px | 4.7m | ✅ |
| 2.0m | 22.9m | 5.00° | +234px | 6.2m | ✅ |
| 2.5m | 28.6m | 5.00° | +234px | 7.8m | ✅ |
| 3.0m | 34.3m | 5.00° | +234px | 9.3m | ✅ |

**重要发现**：采用恒定俯仰角策略后，**所有高度档位的模型顶行均落在相机图像内（+234px）**，完全消除了 PITCH_LIMITS 问题，也不存在任何无效像素行。

#### 与 15m 固定注视距离的对比

| 指标 | 固定 15m 注视 | 恒定 5° 俯仰角 |
|------|------------|-------------|
| H=1.22m（标准） | 14.7m, 4.65° | 13.9m, 5.0°（几乎相同） |
| H=3.0m PITCH_LIMITS | 需扩展至 0.22 rad | **无需修改（0.087 rad）** |
| H=3.0m 无效像素行 | ~20 行（7.7%，天空区域） | **零无效行** |
| 各高度 warp 视觉一致性 | 差（俯仰角差异大） | **高（俯仰角一致）** |
| backbone 微调难度 | 高（不同高度视角差异大） | **低（输入分布相近）** |
| 物理合理性 | 卡车场景不合理 | **符合驾驶经验** |

---

### 2.9.4 近场可见性分析

恒定俯仰角策略下，H=3.0m 的最近可见距离为 9.3m，意味着 X < 9.3m 不在模型视野内。对 LDW/FCW 的影响：

- **LDW**：车道线检测主要工作在 10–192m，9.3m 以内基本不使用 → **无影响**
- **FCW**：目标工作范围 10–100m（卡车场景），9.3m 以内属于极近距离，应急制动已来不及 → **可接受**
- 与 15m 注视距离（最近 6.7m）相比，损失 2.6m 近场 → **对 LDW/FCW 实用价值无显著影响**

对比乘用车（H=1.22m）：最近可见 3.8m，这是因为俯仰角相同但高度更低。卡车 FCW 的实际预警距离应设置在 30–100m，不依赖 < 10m 的近场视觉检测。

---

### 2.9.5 对数据采集和训练的影响

**更新 HEIGHT_CONFIGS**：

```python
import math

TARGET_PITCH_DEG = 5.0

HEIGHT_CONFIGS = {
    'H1': {'height': 1.22, 'look_at': 13.9, 'pitch': -TARGET_PITCH_DEG},
    'H2': {'height': 1.3, 'look_at': 14.9, 'pitch': -TARGET_PITCH_DEG},
    'H3': {'height': 1.5, 'look_at': 17.1, 'pitch': -TARGET_PITCH_DEG},
    'H4': {'height': 2.0, 'look_at': 22.9, 'pitch': -TARGET_PITCH_DEG},
    'H5': {'height': 2.5, 'look_at': 28.6, 'pitch': -TARGET_PITCH_DEG},
    'H6': {'height': 3.0, 'look_at': 34.3, 'pitch': -TARGET_PITCH_DEG},
}
```

**对 backbone 微调的影响**（优化，见 §2.6）：

恒定俯仰角策略使所有高度档位的 warp 图像具有**相同的几何分布**：
- 模型顶行始终对应相机行 +234px（一致的 sky region）
- 地平线（MEDMODEL_CY）始终在相同的相对位置
- 地面纹理的空间分布在不同高度下一致

这意味着：**不同高度的训练样本对 backbone 的冲击更小，仅因地面视角不同（更高处向下看与更低处向下看的透视差），而非 warp 几何的根本性差异**。backbone 微调所需数据量可能比 §2.6 估计的更少。

---

### 2.9.6 运行时行为确认：calibrationd 自动估计 pitch，look_at 无关

通过对源码的分析，确认了以下关键结论：

#### warp 矩阵与 look_at 距离的关系

`calib_from_medmodel` 是一个**固定数学常数**（pitch=0, height=0 硬编码）：

```python
# common/transformations/model.py
medmodel_frame_from_calib_frame = np.dot(medmodel_intrinsics,
  get_view_frame_from_calib_frame(0, 0, 0, 0))   # pitch=0, height=0 硬编码

calib_from_medmodel = np.linalg.inv(medmodel_frame_from_calib_frame[:, :3])
```

`get_warp_matrix()` 是**纯旋转单应矩阵**，不含平移：

```python
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)  # pitch 在这里进入
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix
```

**结论**：
- warp 矩阵**仅依赖 pitch**（通过 `device_from_calib_euler`），**不含高度信息、不含 look_at 距离**
- `look_at_distance` 仅用于 Carla 数据采集时**定义相机朝向**，是辅助工具参数，不进入任何运行时代码

#### calibrationd 自动从视觉里程计估计 pitch

```python
# tools/dashcam/calibrationd.py
observed_rpy = np.array([
    0,
    -np.arctan2(trans[2], trans[0]),   # 从视觉里程计估计 pitch
    np.arctan2(trans[1], trans[0])
])
```

calibrationd **不需要知道 look_at 距离**，而是直接从相机运动（road_transform）的方向向量估计 pitch。若相机以物理 5° 俯仰角安装，calibrationd 将自然收敛到 ~5° 估计值，与训练数据的 warp 几何完全一致。

#### 对恒定 5° 仰角策略的验证

| 环节 | 行为 | look_at 距离的作用 |
|------|------|------------------|
| Carla 数据采集 | 使用 `pitch = 5°`（由 `look_at = H/tan(5°)` 导出） | 定义相机朝向，辅助参数 |
| warp 矩阵计算 | `get_warp_matrix(rpyCalib=[0,0,-5°])` | **无关** |
| calibrationd 运行时 | 从视觉里程计自动收敛到 ~5° | **无关** |
| 模型输入图像 | 与训练时几何一致 | **无关** |

**不同高度（1m~3m）的 look_at 距离差异（11.4m~34.3m）不影响 warp 几何**——只要 pitch 相同，warp 矩阵相同。因此，只要相机以 5° 俯仰角安装，calibrationd 将自动适配，无需额外配置。

---

## 3. 设计决策

### 决策1：是否修改 warp 矩阵？

**结论：不修改 warp 矩阵，添加高度作为显式模型输入。**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 1. 修改 warp 为完整 IPM（逆透视映射） | 地面特征对齐 | 3D 物体（车辆）严重失真；需要精确地面平面估计；推理管线大改 |
| 2. 高度显式输入（HeightConditionedHead） | 轻量；模型可学习高度相关映射；与现有管线兼容 | 依赖高度输入的准确性 |

**选择方案2**：高度嵌入注入 backbone 输出层，模型自适应高度差异。

### 决策2：高度输入的注入方式？

**结论：采用两阶段策略——先验证隐式方案，按需升级为显式注入。**

详细分析见 §2.5。核心逻辑如下：

| 场景 | 推荐方案 |
|------|---------|
| H ≤ 2.0m，对部署简洁性要求高 | 隐式方案（多高度联合训练，无高度输入） |
| H = 2.5–3.0m，或需要任意高度插值 | 显式方案（HeightConditionedHead） |

**第一阶段（隐式基线）**：多高度联合微调 `PretrainedVisionModel`，不修改架构，直接验证隐式方案性能上界。

**第二阶段（按需）**：若隐式方案在 H ≥ 2.0m 精度不足，在其基础上添加显式高度注入：

```
PretrainedVisionModel(backbone, 微调后或冻结)
    → flat: (B, 977)
    → [flat ‖ h_emb(8维)]
    → HeightConditionedHead(MLP)
    → 输出: (B, 977) 与原输出格式兼容
```


---

## 4. 数据采集方案

### 4.1 两阶段采集策略

训练数据采集分为**快速验证**和**正式训练**两个阶段，在资源投入和多样性覆盖之间分级权衡。

| 维度 | 快速验证阶段 | 正式训练阶段 |
|------|------------|------------|
| **目标** | 验证训练流水线、多高度基本效果 | 泛化能力、场景鲁棒性 |
| **高度档位** | H1 + H5（两个极端） | H1–H6 全部 6 档 |
| **地图** | Town04（1 个） | Town04 / Town05 / Town07（3 个） |
| **天气/时间** | ClearNoon（固定） | 3 天气 × 2 时段 |
| **相机姿态** | pitch=5°, yaw=0°（固定） | pitch ∈ [3°,7°]，yaw ∈ [−3°,3°] |
| **帧数** | ~2000 帧（1000/高度） | ~42000 帧（≥5000/高度） |
| **允许过拟合** | ✅ 是（只验证上界） | ❌ 否（需泛化） |

---

### 4.2 高度档位配置

采集 **6 个高度档位**，覆盖主要车型。俯仰角采用**恒定 5° 策略**（详见 §2.9）：

| 档位 | 高度 H (m) | 注视距离 | 名义 pitch | 车型 | 最近可见 X |
|------|-----------|---------|-----------|------|---------|
| H1 | 1.22 | 13.9m | 5.0° | 标准轿车 / comma 3X（基准） | 3.8m |
| H2 | 1.5  | 17.1m | 5.0° | SUV / 越野 | 4.7m |
| H3 | 2.0  | 22.9m | 5.0° | 中型货车 / 厢式面包车 | 6.2m |
| H4 | 2.5  | 28.6m | 5.0° | 大型货车 / 中型卡车 | 7.8m |
| H5 | 3.0  | 34.3m | 5.0° | 重卡 / 长途卡车 | 9.3m |
| H6 | 1.0  | 11.4m | 5.0° | 小型轿车（扩展低端） | 3.1m |

> H1（1.22m）是 openpilot 预训练模型的标准高度，作为微调锚点（§2.6）。H6（1.0m）作为低端扩展，低于标准高度。

---

### 4.3 相机姿态参数设计

#### 4.3.1 Pitch（俯仰角）

实际部署中，相机安装角度因固定支架、吸盘粘贴面、挡风玻璃倾斜度而存在偏差。calibrationd 在运行时自动从视觉里程计收敛到实际 pitch，warp 矩阵随之调整（详见 §2.9.6）。训练数据应覆盖这一分布以保证鲁棒性。

| 参数 | 值 |
|------|-----|
| 名义安装角（恒定策略） | 5.0° |
| 实际安装偏差（典型） | ±1°~±2° |
| 实际安装偏差（最大） | ±3° |
| **训练覆盖范围** | **3° ~ 7°** |
| openpilot PITCH_LIMITS 上限 | 9.74°（0.17 rad） |
| openpilot PITCH_LIMITS 下限 | −5.2°（−0.09074 rad） |

训练 pitch 范围 [3°, 7°] 完全处于 PITCH_LIMITS 内，calibrationd 在部署时可正常收敛。

**快速验证**：固定 pitch = 5.0°
**正式训练**：在 {3°, 4°, 5°, 6°, 7°} 中均匀采样，各高度档位均覆盖全部 pitch 值

#### 4.3.2 Yaw（偏航角）

水平安装偏转同样影响 warp 几何。calibrationd 也会估计 yaw 并纳入 warp 矩阵计算。

| 参数 | 值 |
|------|-----|
| 名义偏航（朝正前方） | 0° |
| 实际安装偏转（典型） | ±1°~±2° |
| 实际安装偏转（最大） | ±4° |
| **训练覆盖范围** | **−3° ~ +3°** |

**快速验证**：固定 yaw = 0°
**正式训练**：在 {−3°, −1.5°, 0°, 1.5°, 3°} 中均匀采样

#### 4.3.3 正式训练姿态矩阵

正式训练每个高度档位需覆盖以下姿态组合：

| pitch \ yaw | −3° | −1.5° | 0° | +1.5° | +3° |
|-------------|-----|-------|-----|-------|-----|
| 3° | ○ | | ○ | | ○ |
| 4° | | ○ | ○ | ○ | |
| **5°（名义）** | ○ | ○ | **●** | ○ | ○ |
| 6° | | ○ | ○ | ○ | |
| 7° | ○ | | ○ | | ○ |

> ● 名义中心点，帧数权重 2×；○ 覆盖点，帧数权重 1×。中心点权重加倍确保名义安装条件的训练密度。

---

### 4.4 地图与场景条件

#### 4.4.1 地图选择

| 地图 | 道路特征 | 用途 |
|------|---------|------|
| **Town04** | 高速公路 + 乡村双车道，车道线清晰，弯道少 | 快速验证 **+** 正式训练 |
| **Town05** | 城区网格路网，多车道 + 十字路口 | 正式训练 |
| **Town07** | 乡村小道，弯道多，无中心线 | 正式训练（边缘场景） |

Town04 是快速验证的首选：车道线规则清晰，便于定性检查模型输出。

#### 4.4.2 天气与时段

| 条件代码 | 描述 | 阶段 |
|---------|------|------|
| `ClearNoon` | 晴天正午，高对比度 | 验证 + 正式 |
| `CloudyNoon` | 多云正午，漫射光 | 正式 |
| `MidRainyNoon` | 中雨，路面反光 | 正式 |
| `ClearSunset` | 晴天黄昏，低角度侧光 | 正式 |

快速验证阶段只用 `ClearNoon`，条件最稳定，排除光照干扰。

正式训练：**3 天气（Clear/Cloudy/Rain）× 2 时段（Noon/Sunset）= 6 个场景条件**。

#### 4.4.3 其他场景参数

| 参数 | 值 | 说明 |
|------|-----|------|
| NPC 车辆密度 | 40–60 辆 | 保证有前车目标供 FCW 训练 |
| 自车速度 | 40–100 km/h | 模拟高速/城区混合工况 |
| 路线类型 | 直道 + 弯道 + 匝道 | 避免单一场景过拟合 |

---

### 4.5 采集规模与分配

#### 快速验证阶段（总计约 2000 帧）

| 高度 | 地图 | 场景 | pitch | yaw | 帧数 |
|------|------|------|-------|-----|------|
| H1 (1.22m) | Town04 | ClearNoon | 5° | 0° | 1000 |
| H5 (3.0m) | Town04 | ClearNoon | 5° | 0° | 1000 |

目标：2000 帧在 ~30 分钟内完成采集，1 小时内完成预处理 + 训练试跑。

#### 正式训练阶段（总计约 42000 帧）

每个高度档位的场景分配（以 H3–H5 为例，高偏移档位分配更多帧数）：

| 高度 | 帧数（总） | 地图分配 | 场景多样性 | pitch×yaw 覆盖 |
|------|---------|---------|---------|--------------|
| H1 (1.22m) | 3000 | Town04×3 | Clear+Cloudy+Rain | 3×3=9种 |
| H2 (1.5m) | 4000 | 3地图 | 4种 | 5×5=15种（采样） |
| H3 (2.0m) | 6000 | 3地图 | 6种 | 5×5=15种 |
| H4 (2.5m) | 7000 | 3地图 | 6种 | 5×5=15种 |
| H5 (3.0m) | 10000 | 3地图 | 6种 | 5×5=15种 |
| H6 (1.0m) | 4000 | Town04+Town05 | 4种 | 3×3=9种 |
| **合计** | **~34000** | | | |

> H5（3.0m）帧数最多，因分布偏移最大（§2.6.3）且是最具挑战性的场景。H1 因与预训练分布一致，帧数最少。

**追加采集缓冲**：建议在每个高度多采集 20%（约 +7000 帧），用于剔除异常帧（碰撞、急停、场景切换）。最终有效帧目标 ~34000，原始采集目标约 **42000 帧**。

---

### 4.6 Carla 配置代码

#### HEIGHT_CONFIGS（含车型映射）

```python
import math

TARGET_PITCH_DEG = 5.0

HEIGHT_CONFIGS = {
    'H1': {'height': 1.22, 'look_at': 13.9, 'vehicle': 'vehicle.toyota.prius'},
    'H2': {'height': 1.5,  'look_at': 17.1, 'vehicle': 'vehicle.ford.mustang'},
    'H3': {'height': 2.0,  'look_at': 22.9, 'vehicle': 'vehicle.mercedes.sprinter'},
    'H4': {'height': 2.5,  'look_at': 28.6, 'vehicle': 'vehicle.carlamotors.firetruck'},
    'H5': {'height': 3.0,  'look_at': 34.3, 'vehicle': 'vehicle.carlamotors.european_hgv'},
    'H6': {'height': 1.0,  'look_at': 11.4, 'vehicle': 'vehicle.tesla.model3'},
}

# 姿态扰动配置
PITCH_VARIANTS = [3.0, 4.0, 5.0, 6.0, 7.0]   # 度，向下为正
YAW_VARIANTS   = [-3.0, -1.5, 0.0, 1.5, 3.0]  # 度，右偏为正

# 快速验证：单一姿态
QUICK_POSE = {'pitch': -5.0, 'yaw': 0.0}
```

#### 场景条件配置

```python
# 正式训练场景矩阵
SCENE_CONFIGS_FULL = [
    {'map': 'Town04', 'weather': 'ClearNoon'},
    {'map': 'Town04', 'weather': 'CloudyNoon'},
    {'map': 'Town04', 'weather': 'MidRainyNoon'},
    {'map': 'Town05', 'weather': 'ClearNoon'},
    {'map': 'Town05', 'weather': 'ClearSunset'},
    {'map': 'Town07', 'weather': 'ClearNoon'},
]

# 快速验证：仅使用第一条
SCENE_CONFIGS_QUICK = SCENE_CONFIGS_FULL[:1]
```

#### CameraConfig（含 yaw 支持）

```python
# tools/dashcam/carla_world.py
class CameraConfig:
    height: float = 1.22    # 相机离地高度（米）
    pitch: float = -5.0     # 相机俯仰角（度，负值=向下）
    yaw:   float = 0.0      # 相机偏航角（度，正值=右偏）
    forward_offset: float = 2.0  # 前向偏移（米）
```

#### 批量采集入口

```python
def collect_batch(phase='quick', output_base='data/multi_height'):
    """
    phase='quick': 快速验证，固定条件，H1+H5
    phase='full':  正式训练，全高度，多场景，多姿态
    """
    heights = ['H1', 'H5'] if phase == 'quick' else list(HEIGHT_CONFIGS.keys())
    scenes  = SCENE_CONFIGS_QUICK if phase == 'quick' else SCENE_CONFIGS_FULL
    pitches = [QUICK_POSE['pitch']] if phase == 'quick' else [-p for p in PITCH_VARIANTS]
    yaws    = [QUICK_POSE['yaw']]   if phase == 'quick' else YAW_VARIANTS
    frames  = 1000 if phase == 'quick' else 800  # 每个条件组合的帧数

    for h_key in heights:
        h = HEIGHT_CONFIGS[h_key]
        for scene in scenes:
            for pitch in pitches:
                for yaw in yaws:
                    tag = f"{h_key}_p{abs(pitch):.0f}_y{yaw:+.0f}_{scene['map']}_{scene['weather']}"
                    run_collection(
                        camera_height=h['height'],
                        camera_pitch=pitch,
                        camera_yaw=yaw,
                        vehicle=h['vehicle'],
                        map_name=scene['map'],
                        weather=scene['weather'],
                        n_frames=frames,
                        output_dir=f"{output_base}/{tag}",
                    )
```

---

## 5. 数据标注方案

### 5.1 modeld 在线标注（H ≤ 1.5m）

对于接近标准高度的数据，使用 `tools/dashcam/run.py --record-modeld` 直接采集 modeld 输出作为标注：

```bash
python tools/dashcam/run.py \
    --camera-height 1.3 \
    --record-modeld \
    --output-dir data/multi_height/H1.3
```

## 6. 模型训练方案

### 6.1 架构设计：高度条件化模型

#### HeightConditionedModel 结构

```
输入：
  img:     (B, 12, 128, 256) uint8   — road camera
  big_img: (B, 12, 128, 256) uint8   — wide camera
  height:  (B, 1)            float32 — 相机安装高度（米）

处理流程：
  1. backbone(img, big_img) → flat (B, 977)    [backbone冻结]
  2. height_embedding(height) → h_emb (B, 8)   [可训练]
  3. concat([flat, h_emb]) → (B, 985)
  4. conditioned_head(985 → 977) → out (B, 977) [可训练]

输出：与 PretrainedVisionModel 完全兼容的字典格式
```

#### 高度嵌入设计

```python
class HeightEmbedding(nn.Module):
    """将标量高度编码为连续嵌入向量。

    使用正弦位置编码风格，对高度变化敏感。
    """
    def __init__(self, embed_dim=8, height_min=0.8, height_max=3.5):
        super().__init__()
        self.height_min = height_min
        self.height_max = height_max
        # 可学习的频率参数
        self.freq = nn.Parameter(torch.randn(embed_dim // 2) * 0.1)
        self.phase = nn.Parameter(torch.zeros(embed_dim // 2))

    def forward(self, height):  # height: (B, 1)
        h_norm = (height - self.height_min) / (self.height_max - self.height_min)  # [0, 1]
        angles = h_norm * self.freq.unsqueeze(0) + self.phase.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 8)
```

#### 高度条件化头部

```python
class HeightConditionedHead(nn.Module):
    """轻量级高度条件化 MLP，调制 backbone 输出。"""
    def __init__(self, feat_dim=977, height_embed_dim=8, hidden_dim=256):
        super().__init__()
        self.height_emb = HeightEmbedding(embed_dim=height_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + height_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        # 初始化为恒等映射（残差），保证初始时接近预训练模型输出
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, flat, height):
        h_emb = self.height_emb(height)
        x = torch.cat([flat, h_emb], dim=-1)
        return flat + self.mlp(x)  # 残差连接
```

#### 完整模型

```python
class HeightConditionedVisionModel(nn.Module):
    """高度条件化驾驶视觉模型。

    backbone: PretrainedVisionModel（openpilot 预训练，默认冻结）
    head:     HeightConditionedHead（轻量级，随机初始化）
    """
    def __init__(self, onnx_path, freeze_backbone=True):
        super().__init__()
        self.backbone = PretrainedVisionModel(onnx_path, freeze=freeze_backbone)
        self.head = HeightConditionedHead()

    def forward(self, img, big_img, height):
        # 1. 预训练 backbone 提取特征
        backbone_out = self.backbone(img, big_img)  # dict(977维)
        flat = torch.cat([backbone_out[k] for k in OUTPUT_NAMES], dim=-1)  # (B, 977)

        # 2. 高度条件化调制
        flat_conditioned = self.head(flat, height)  # (B, 977)

        # 3. 重新分割为输出字典
        return split_output_dict(flat_conditioned)  # dict，与 backbone_out 格式相同

    def freeze_backbone(self):
        self.backbone.freeze_backbone()

    def unfreeze_all(self):
        self.backbone.unfreeze_all()

    def n_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**参数量对比**：
- backbone（冻结）：~23M 参数，0 可训练
- HeightConditionedHead：~250K 参数（<1%），全部可训练
- 总训练参数量：~250K（极轻量）

### 6.2 数据准备

#### 预处理缓存扩展

现有 `preprocess_cache.py` 生成的 NPZ 缓存需增加 `camera_height` 字段：

```python
# preprocess_cache.py — 新增字段
cache_dict = {
    'road_yuv': road_yuv,    # (12, 128, 256) uint8
    'wide_yuv': wide_yuv,    # (12, 128, 256) uint8
    'camera_height': np.float32(frame_data['camera_height']),  # 标量
    # ... 其他标签字段不变
}
```

#### 多高度数据集混合策略

```python
class MultiHeightDataset(Dataset):
    """按高度均匀采样的多高度数据集。

    策略：每个 batch 中各高度档位均匀出现，避免高度分布偏差。
    """
    def __init__(self, height_cache_dirs: dict[float, str]):
        # height_cache_dirs: {1.0: 'data/H1.0_cache', 1.3: '...', ...}
        self.datasets = {
            h: CachedDualCameraDrivingDataset(d)
            for h, d in height_cache_dirs.items()
        }
        # 按最小数据集大小截断，确保均匀采样
        min_size = min(len(d) for d in self.datasets.values())
        self.size = min_size * len(self.datasets)

    def __getitem__(self, idx):
        heights = sorted(self.datasets.keys())
        h = heights[idx % len(heights)]
        local_idx = idx // len(heights)
        sample = self.datasets[h][local_idx]
        sample['camera_height'] = torch.tensor([h], dtype=torch.float32)
        return sample
```

#### 数据增强：高度 jitter

```python
# 训练时对 camera_height 添加轻微扰动，增强鲁棒性
height_jitter = 0.1  # ±0.1m
height = height + torch.randn_like(height) * height_jitter
height = height.clamp(0.8, 3.5)
```

### 6.3 训练流程

#### 阶段0：零样本基线评估（无需训练）

在各高度数据集上直接运行标准 `PretrainedVisionModel`（不传高度），量化精度退化：

```bash
python tools/dashcam/train/evaluate.py \
    --model pretrained \
    --onnx-path selfdrive/modeld/models/driving_vision.onnx \
    --cache-dirs data/multi_height/H1.0_cache data/multi_height/H3.0_cache \
    --output baseline_report.json
```

#### 阶段1：高度头微调（backbone 冻结）

```bash
python tools/dashcam/train/train.py \
    --model height-conditioned \
    --onnx-path selfdrive/modeld/models/driving_vision.onnx \
    --freeze-backbone \
    --height-cache-dirs data/multi_height/H*_cache \
    --output-dir checkpoints/height_v1 \
    --epochs 50 --batch-size 8 --lr 1e-4 \
    --early-stop 15
```

预期效果：轻量头部快速收敛（~10 epoch），backbone 知识完全保留。

#### 阶段2（可选）：全网络微调

当阶段1收敛后，解冻 backbone 进行微调：

```bash
python tools/dashcam/train/train.py \
    --model height-conditioned \
    --resume checkpoints/height_v1/best.pt \
    --unfreeze-backbone \
    --height-cache-dirs data/multi_height/H*_cache \
    --output-dir checkpoints/height_v2 \
    --epochs 30 --batch-size 4 --lr 1e-5 \
    --early-stop 10
```

注意：全网络微调需要更小学习率（1e-5），防止预训练权重被过度修改。

### 6.4 损失函数

现有 `DrivingLoss`（GaussianNLL）不需修改，直接复用：

```python
# losses.py — 现有 DrivingLoss 已支持所有输出头
# 按高度分层监控 loss，用于验证各高度档位的收敛情况

def compute_height_stratified_loss(preds, targets, heights):
    """按高度档位分别统计 loss，用于 tensorboard 监控。"""
    height_bins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for i in range(len(height_bins) - 1):
        mask = (heights >= height_bins[i]) & (heights < height_bins[i+1])
        if mask.any():
            loss_i = compute_driving_loss(preds[mask], targets[mask])
            log_metric(f'val_loss_H{height_bins[i]:.1f}', loss_i)
```

---

## 7. 评价方案

### 7.1 离线评价指标

#### 车道线评价

模型输出 `lane_lines (4, 33, 2)` 中的横向坐标与 GT 对比：

```python
# 按 X 距离分段的 MAE
X_IDXS = ModelConstants.X_IDXS  # [0, ..., 192m]，33个点

for segment, (x_min, x_max) in {'near': (0, 30), 'mid': (30, 80), 'far': (80, 192)}.items():
    mask = [(x_min <= x < x_max) for x in X_IDXS]
    lane_mae = mean_absolute_error(pred_ll[:, mask, 0], gt_ll[:, mask, 0])  # 横向 y
    print(f"Lane MAE [{segment}]: {lane_mae:.3f} m")
```

#### 前车检测评价

```python
# BEV mAP，IOU > 0.5（投影到 X-Y 平面）
def compute_lead_map(pred_lead, gt_lead, iou_threshold=0.5):
    """计算前车检测 mAP（BEV，基于预测框与 GT 框的 IOU）。"""
    # 以 lead[best_prob].x, lead[best_prob].y 为框中心，车辆尺寸为 4.5×2.0m
    ...
```

#### Pose 误差

```python
# 平移/旋转均方误差
pose_trans_rmse = torch.sqrt(((pred_pose[:, :3] - gt_pose[:, :3])**2).mean())
pose_rot_rmse   = torch.sqrt(((pred_pose[:, 3:] - gt_pose[:, 3:])**2).mean())
```

### 7.2 分层评价设计

#### 按高度分层

| 高度档位 | 数据集 | 评价角色 |
|---------|--------|---------|
| H1 (1.22m) | 采集 200 帧 | 测试集（openpilot 标准高度，验证不退化） |
| H2 (1.3m) | 采集 500 帧 | 测试集（标准高度，基线参考） |
| H3 (1.5m) | 训练集 | 训练 |
| H4 (2.0m) | 训练集 | 训练 |
| H5 (2.5m) | 测试集（留出） | 跨高度泛化测试 |
| H6 (3.0m) | 训练集 | 训练 |

> H2.5 作为泛化测试集（训练中不出现），验证模型在未见高度档位的插值能力。

#### 跨高度泛化矩阵

| 训练集 \ 测试集 | H1 | H2 | H3 | H4 | **H5** | H6 |
|---------------|----|----|----|----|--------|-----|
| 全高度联合 | ✓ | ✓ | ✓ | ✓ | **泛化测试** | ✓ |
| 仅 H2 (基线) | 基线 | 基线 | 退化 | 退化 | 退化 | 退化 |

### 7.3 在线评价（Carla 实时仿真）

在 Carla 中各高度下运行完整感知管线：

```bash
# 各高度在线评价
for height in 1.0 1.3 1.5 2.0 2.5 3.0; do
    python tools/dashcam/run.py \
        --camera-height $height \
        --custom-modeld checkpoints/height_v2/best.pt \
        --eval-mode \
        --output-dir eval/online/H$height
done
```

在线评价指标：
- **FPS**：推理帧率（目标 ≥ 20 FPS）
- **车道线稳定性**：连续帧间横向偏差标准差
- **前车检测率**：在 NPC 车辆存在时的召回率
- **主观评价**：目视检查可视化输出

与标准 openpilot modeld 的对比（相同 H2=1.3m 场景）：

```bash
python tools/dashcam/run.py --camera-height 1.3 --use-stock-modeld  # 对照
python tools/dashcam/run.py --camera-height 1.3 --custom-modeld ...  # 实验
```

---

## 8. 实施路线图

| 阶段 | 任务 | 关键产出 | 依赖 |
|------|------|---------|------|
| **Phase 0** | 零样本基线评估 | 各高度精度退化量化报告 | 无（直接用预训练模型） |
| **Phase 1** | 多高度数据采集 | 6×5000帧 NPZ 数据集，含 GT 标注 | carla_world.py 高度参数支持 |
| **Phase 2** | Carla GT 标注脚本 | `carla_gt_extractor.py`（车道线+前车） | Carla waypoint API |
| **Phase 3** | 数据预处理 | 多高度缓存（`preprocess_cache.py` 扩展） | Phase 1 数据 |
| **Phase 4** | 模型训练 | `HeightConditionedVisionModel` 权重 | Phase 3 缓存 |
| **Phase 5** | 离线评价 | 高度分层评价报告（各指标） | Phase 4 权重 |
| **Phase 6** | 在线验证 | Carla 实时仿真 FPS + 精度 | Phase 5 权重 |
| **Phase 7** | 迭代优化 | 最终模型权重，评价报告 | Phase 6 反馈 |

### 时间估计

- Phase 0（零样本评估）：半天（运行评估脚本）
- Phase 1–2（数据采集与标注）：2–3 天（Carla 采集 + GT 脚本开发）
- Phase 3（预处理缓存）：0.5 天（复用现有 preprocess_cache.py）
- Phase 4（模型训练，阶段1）：~1 天（50 epoch × ~20s/epoch = ~17min，GPU 加速）
- Phase 5–6（评价）：1 天
- Phase 7（迭代）：视反馈而定

---

## 9. 关键文件索引

| 文件 | 用途 | 需修改 |
|------|------|--------|
| `tools/dashcam/carla_world.py` | 相机挂载（需多高度批量采集支持） | **是** |
| `tools/dashcam/dual_data_recorder.py` | 数据格式（`camera_height` 已有） | 否 |
| `tools/dashcam/train/pretrained_model.py` | backbone（`PretrainedVisionModel`） | 否（新建子类） |
| `tools/dashcam/train/train.py` | 训练主脚本 | **是**（支持 height 输入） |
| `tools/dashcam/train/dataset.py` | 数据集（需读取 `camera_height`） | **是** |
| `tools/dashcam/train/losses.py` | 损失函数（现有，无需改动） | 否 |
| `tools/dashcam/train/preprocess_cache.py` | 预处理缓存（需存 `camera_height`） | **是** |
| `common/transformations/model.py` | `get_warp_matrix`（理解高度效应） | 否 |
| `selfdrive/modeld/constants.py` | `X_IDXS`，模型输出维度常量 | 否 |
| `tools/dashcam/calibrationd.py` | 高度在线估计（`height` 字段） | 否（参考） |

新增文件：
- `tools/dashcam/train/height_conditioned_model.py` — `HeightConditionedVisionModel` 实现
- `tools/dashcam/carla_gt_extractor.py` — Carla GT 车道线/前车标注脚本
- `tools/dashcam/train/evaluate_height.py` — 多高度离线评价脚本

---

## 附录A：高度误差精确计算

对于安装高度 H 的相机，观察距离 X 处的地面点：

**投影误差（弧度）**：

```
δθ = arctan((H - H₀) / X)
其中 H₀ = 1.22m（标准高度）
```

**像素偏移**（焦距 f = 910px）：

```
δy_px = f × tan(δθ) ≈ f × (H - H₀) / X
```

**完整误差表**（ΔH = H - 1.22m）：

| H (m) | ΔH (m) | X=10m | X=20m | X=30m | X=50m | X=100m |
|-------|--------|-------|-------|-------|-------|--------|
| 1.0   | -0.22  | -20px | -10px | -7px  | -4px  | -2px   |
| 1.5   | +0.28  | +25px | +13px | +8px  | +5px  | +3px   |
| 2.0   | +0.78  | +71px | +35px | +24px | +14px | +7px   |
| 2.5   | +1.28  | +116px| +58px | +39px | +23px | +12px  |
| 3.0   | +1.78  | +162px| +81px | +54px | +32px | +16px  |

1px ≈ 0.06° ≈ 地面约 0.01m（X=10m 处），→ 远场误差对车道线预测影响有限，近场误差需重点解决。

---

## 附录B：模型输出格式参考

```python
# ONNX 输出切片（来自 pretrained_model.py）
ONNX_OUTPUT_SLICES = {
    'pose':                   slice(87,   99),   # 12 dims (6 mean + 6 log_sigma)
    'wide_from_device_euler': slice(99,   105),  # 6 dims  (3 mean + 3 log_sigma)
    'road_transform':         slice(105,  117),  # 12 dims (6 mean + 6 log_sigma)
    'lane_lines':             slice(117,  645),  # 528 dims (4 lanes × 33 pts × 4 MDN)
    'lane_lines_prob':        slice(645,  653),  # 8 dims
    'road_edges':             slice(653,  917),  # 264 dims (2 edges × 33 pts × 4 MDN)
    'lead':                   slice(917,  1061), # 144 dims (3×6×8 MDN)
    'lead_prob':              slice(1061, 1064), # 3 dims
}

# lane_lines MDN 格式（parse_mdn 全均值在前）
# lane_lines (528) → reshape(4, 33, 4) 后：
#   [:2] → mean (y_lat, z_height)
#   [2:] → log_sigma (y_lat, z_height)

# ModelConstants.X_IDXS: 33个距离点，0–192m，平方间隔
# X_IDXS[0] = 0m, X_IDXS[16] ≈ 48m, X_IDXS[32] = 192m
```
