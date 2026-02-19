# openpilot 视觉神经网络架构深度分析

> 基于 ONNX 模型逆向拓扑追踪与源码阅读的完整架构还原

---

## 目录

- [第1章：概述](#第1章概述)
- [第2章：输入预处理管线](#第2章输入预处理管线)
- [第3章：视觉网络 Backbone](#第3章视觉网络-backbone)
- [第4章：输出头架构](#第4章输出头架构)
- [第5章：输出语义与解析](#第5章输出语义与解析)
- [第6章：策略网络](#第6章策略网络)
- [第7章：端到端推理流程](#第7章端到端推理流程)
- [第8章：架构设计特征总结](#第8章架构设计特征总结)
- [附录A：源码文件索引表](#附录a源码文件索引表)
- [附录B：算子类型统计表](#附录b算子类型统计表)
- [附录C：完整卷积操作清单](#附录c完整卷积操作清单)

---

## 第1章：概述

### 1.1 双网络架构总览

openpilot 的驾驶模型采用**双网络串联架构**，将感知与决策解耦为两个独立的 ONNX 模型：

| 属性 | 视觉网络 (Vision) | 策略网络 (Policy) |
|------|-------------------|-------------------|
| 文件 | `driving_vision.onnx` | `driving_policy.onnx` |
| 大小 | 45 MB (fp16) | 14 MB (fp16) |
| 参数量 | 23,057,880 (23.1M) | 6,949,495 (6.9M) |
| ONNX 节点数 | 496 | 102 |
| 算子类型数 | 19 | 18 |
| 输入 | 双相机图像 (uint8) | 特征缓冲 + desire + 交通惯例 |
| 输出 | 1576D (fp16) | 1000D (fp16) |
| 职责 | 图像特征提取 + 感知输出 | 时序建模 + 路径规划 |

### 1.2 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        openpilot 驾驶模型数据流                         │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ camerad   │───>│ VisionIPC│───>│ OpenCL 预处理 │───>│  视觉网络     │  │
│  │ (相机帧)  │    │ (共享内存)│    │ (warp+YUV)   │    │  (23.1M)     │  │
│  └──────────┘    └──────────┘    └──────────────┘    └──────┬───────┘  │
│                                                              │          │
│       ┌──────────────────────────────────────────────────────┤          │
│       │  hidden_state (512D)      感知输出 (1064D)           │          │
│       │  ↓ 存入时序缓冲            ↓ lane/lead/meta 等      │          │
│       │                                                      │          │
│  ┌────▼──────┐   desire_pulse (8D)                           │          │
│  │ 策略网络   │<── traffic_convention (2D)                    │          │
│  │ (6.9M)    │<── features_buffer (25×512D)                  │          │
│  └─────┬─────┘                                               │          │
│        │                                                     │          │
│        ▼ plan (990D) + desire_state (8D)                     │          │
│  ┌───────────┐    ┌────────────┐    ┌────────────────────┐   │          │
│  │  解析输出  │───>│ 动作计算    │───>│ cereal 消息发布     │   │          │
│  │ (MDN/MHP) │    │ (曲率/加速) │    │ (modelV2/odom)     │   │          │
│  └───────────┘    └────────────┘    └────────────────────┘   │          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 ONNX 模型统计摘要

视觉网络 496 个节点分布在 19 种算子类型中，其中 Mul(128) 最多——主要来自 GELU tanh 近似的 6 步展开。策略网络 102 个节点分布在 18 种算子类型中，使用了 LayerNormalization、MatMul 等 Transformer 特有算子。

---

## 第2章：输入预处理管线

### 2.1 相机硬件配置

openpilot 运行在 comma 3X 硬件上，配备双路相机：

| 相机 | 传感器 | 分辨率 | 焦距 | 视场角 |
|------|--------|--------|------|--------|
| 前置窄角 (fcam) | AR0231 | 1928×1208 | 2648.0 | ~45° |
| 前置广角 (ecam) | AR0231 | 1928×1208 | 567.0 | ~120° |

> 源码参考：`common/transformations/camera.py:51`

```python
_ar_ox_config = DeviceCameraConfig(
    CameraConfig(1928, 1208, 2648.0),  # fcam
    _ar_ox_fisheye,                     # dcam (driver)
    _ar_ox_fisheye                      # ecam (广角, focal=567.0)
)
```

不同硬件/传感器组合通过 `DEVICE_CAMERAS` 字典映射，支持 AR0231、OX03C10、OS04C10 等传感器。

### 2.2 坐标系定义

openpilot 使用三个关键坐标系：

| 坐标系 | 轴定义 | 用途 |
|--------|--------|------|
| device (设备) | x→前, y→右, z→下 | 硬件安装坐标 |
| view (视图) | x→右, y→下, z→前 | 相机成像坐标 |
| calib (标定) | 经标定校正后的设备坐标 | 补偿安装偏差 |

坐标系转换矩阵（`camera.py:75-80`）：

```python
device_frame_from_view_frame = np.array([
    [0., 0., 1.],   # device_x = view_z (forward)
    [1., 0., 0.],   # device_y = view_x (right)
    [0., 1., 0.]    # device_z = view_y (down)
])
view_frame_from_device_frame = device_frame_from_view_frame.T
```

### 2.3 Warp Matrix（透视变换矩阵）

模型需要将真实相机图像变换到**虚拟相机坐标系**。变换链如下：

```
model_frame ←── calib_frame ←── device_frame ←── view_frame ←── camera_frame
    ↑                 ↑               ↑               ↑
 model_intrinsics  rot_from_euler  view_from_device  cam_intrinsics
```

核心函数（`common/transformations/model.py:65-70`）：

```python
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix
```

变换矩阵的计算融合了**在线标定角度**（来自 `liveCalibration` 的 rpyCalib），实时补偿设备安装偏差。

在 `modeld.py:434-436` 中，主流使用 MedModel 变换，extra 流使用 SBIGModel 变换：

```python
model_transform_main = get_warp_matrix(device_from_calib_euler,
    dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False)
model_transform_extra = get_warp_matrix(device_from_calib_euler,
    dc.ecam.intrinsics, True)
```

### 2.4 MedModel 虚拟相机坐标系

视觉网络使用两个虚拟相机参数，分别对应主/额外输入：

| 参数 | MedModel (主路) | SBIGModel (额外路) |
|------|-----------------|---------------------|
| 输入分辨率 | 512×256 | 512×256 |
| 焦距 | 910.0 | 455.0 |
| 光心 CX | 256.0 | 256.0 |
| 光心 CY | 47.6 | 0.5×(256+47.6)=151.8 |

> 源码参考：`common/transformations/model.py:10-40`

MedModel 的 CY=47.6 偏离中心，意味着虚拟相机的光轴略微朝上偏移，这使得模型能看到**更多的道路前方区域**而非天空。SBIGModel 焦距减半（455 vs 910），覆盖更广的视野。

### 2.5 OpenCL 图像预处理

图像预处理通过两个 OpenCL kernel 完成：

**第一步：透视变换（`transform.cl`）**

`warpPerspective` kernel 对 NV12 格式的 Y/U/V 通道分别进行 3×3 齐次透视变换，输出 512×256 的 Y/U/V 分量。变换使用**双线性插值**：

```c
// 透视变换 + 双线性插值（核心逻辑）
float X0 = M[0]*dx + M[1]*dy + M[2];
float Y0 = M[3]*dx + M[4]*dy + M[5];
float W  = M[6]*dx + M[7]*dy + M[8];
W = W != 0.0f ? INTER_TAB_SIZE / W : 0.0f;
int X = rint(X0 * W), Y = rint(Y0 * W);
// ... 4邻域双线性插值
```

**第二步：YUV 重组（`loadyuv.cl`）**

`loadys` kernel 将 Y 通道按 2×2 子采样重组为 4 个通道：

```
原始 Y (512×256) → 4个通道，每通道 256×128:
  ch0: Y[::2, ::2]    (偶行偶列)
  ch1: Y[::2, 1::2]   (偶行奇列)
  ch2: Y[1::2, ::2]   (奇行偶列)
  ch3: Y[1::2, 1::2]  (奇行奇列)
ch4: U (256×128)
ch5: V (256×128)
```

最终每帧图像编码为 **6 通道 × 128 × 256** 的 uint8 张量。

### 2.6 双帧时序输入

视觉网络使用**帧拼接**方式引入短时序信息：

| 参数 | 值 | 源码 |
|------|----|------|
| N_FRAMES | 2 | `constants.py:16` |
| temporal_skip | 4 (=20Hz/5Hz) | `commonmodel.cc:10` |
| 模型运行频率 | 20 Hz | `constants.py:17` |
| 上下文频率 | 5 Hz | `constants.py:18` |

在 C++ 层（`commonmodel.cc:23-30`），`DrivingModelFrame::prepare()` 维护一个 (temporal_skip+1) 帧的循环缓冲。每次推理取**当前帧**和 **temporal_skip=4 帧前的帧**（200ms 间隔），拼接为 2×6=**12 通道**：

```cpp
// 时间帧滑动窗口
for (int i = 0; i < temporal_skip; i++) {
    clEnqueueCopyBuffer(q, img_buffer_20hz_cl, img_buffer_20hz_cl,
        (i+1)*frame_size_bytes, i*frame_size_bytes, frame_size_bytes, ...);
}
loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, last_img_cl);
// 拼接旧帧 + 新帧 → 12 通道
copy_queue(&loadyuv, q, img_buffer_20hz_cl, input_frames_cl, 0, 0, frame_size_bytes);
copy_queue(&loadyuv, q, last_img_cl, input_frames_cl, 0, frame_size_bytes, frame_size_bytes);
```

### 2.7 双目拼接

两路相机流在进入网络前进行**通道维拼接**：

```
img:     [1, 12, 128, 256]  ← 主路（窄角/广角，取决于硬件配置）
big_img: [1, 12, 128, 256]  ← 广角路

Cast(uint8→fp16) → Concat(axis=1) → [1, 24, 128, 256]
```

> ONNX 节点 [0-2]：Cast img → Cast big_img → Concat

在 `modeld.py:460` 中，`big_img` 始终来自广角相机，`img` 来自主流（取决于硬件是否有双路）：

```python
bufs = {name: buf_extra if 'big' in name else buf_main
        for name in model.vision_input_names}
```

---

## 第3章：视觉网络 Backbone

### 3.1 输入归一化

网络的第一步是将 uint8 图像转为浮点并做**逐通道减均值除标准差**归一化（ONNX 节点 [3-4]）：

```
Cast(uint8→fp16) → Sub(mean[1,24,1,1]) → Div(std[1,24,1,1])
```

值得注意的是：
- 均值和标准差作为**固定权重**存储在 ONNX initializer 中（非 BatchNorm 的运行统计）
- 整个网络**不使用 BatchNorm 或 LayerNorm**（backbone 部分），这是与标准 ConvNeXt 的显著差异

### 3.2 Stem（初始下采样器）

Stem 由 3 个卷积层组成，将空间分辨率从 128×256 降至 32×64：

| 层 | 类型 | 权重 | stride | 输出形状 | 激活 |
|----|------|------|--------|----------|------|
| stem.0 | Conv 3×3 | [64, 24, 3, 3] | 2 | [1, 64, 64, 128] | GELU |
| stem.1 | DWConv 3×3 | [64, 1, 3, 3] g=64 | 2 | [1, 64, 32, 64] | GELU |
| stem.2 | Conv 1×1 | [64, 64, 1, 1] | 1 | [1, 64, 32, 64] | GELU |

Stem 快速将 24 通道输入映射到 64 通道，并以 4× 下采样进入 backbone。

### 3.3 四阶段 Backbone 总览

| Stage | 通道 | Block 数 | 空间分辨率 | MLP expand | 参数 |
|-------|------|----------|-----------|------------|------|
| 0 | 64 | 2 | 32×64 | 64→192→64 | ~0.5M |
| 1 | 128 | 2 | 16×32 | 128→384→128 | ~1.2M |
| 2 | 256 | 6 | 8×16 | 256→768→256 | ~6.0M |
| 3 | 512 | 2 | 4×8 | 512→1536→512 | ~8.5M |

各 stage 之间通过**下采样器**（DWConv 7×7 stride=2 + Conv 1×1 + GELU）过渡。

### 3.4 ConvNeXt Block 详细结构

每个 ConvNeXt Block 包含**token_mixer** + **MLP** 两部分，与标准 ConvNeXt 相比有显著差异：

```
              ┌──────────────────────────────────────┐
              │          ConvNeXt Block               │
              │                                        │
  input ─→ DWConv 3×3 (token_mixer) ─→ x            │
              │                           │            │
              │    DWConv 7×7 (MLP spatial)            │
              │         │                              │
              │    Conv 1×1 fc1 (expand ×3)            │
              │         │                              │
              │       GELU                             │
              │         │                              │
              │    Conv 1×1 fc2 (project)              │
              │         │                              │
              │    LayerScale (γ)                      │
              │         │                              │
              │     x + LayerScale ─→ output           │
              └──────────────────────────────────────┘
```

**关键观察**：
1. **残差连接**包裹的是 `MLP(x)` 部分，而**不包含** token_mixer。即：`output = token_mixer(input) + LayerScale(MLP(token_mixer(input)))`
2. **双尺度 DWConv**：先 3×3（局部特征混合），再 7×7（更大感受野），两者串行无激活
3. **无 Normalization**：标准 ConvNeXt 在 DWConv 后使用 LayerNorm，此处完全省略

以 Stage 0 Block 0 为例（ONNX 节点 [47-65]）的 PyTorch 伪代码：

```python
def forward(self, input):
    # Token Mixer
    x = self.token_mixer(input)      # DWConv3x3(64, g=64) [节点47]
    # MLP
    h = self.mlp_conv(x)             # DWConv7x7(64, g=64) [节点48]
    h = self.fc1(h)                  # Conv1x1(64→192)    [节点49]
    h = gelu(h)                      # GELU tanh近似      [节点50-62]
    h = self.fc2(h)                  # Conv1x1(192→64)    [节点63]
    h = self.layer_scale.gamma * h   # LayerScale         [节点64]
    return x + h                     # 残差连接            [节点65]
```

### 3.5 Stage 间下采样器

相邻 stage 之间通过下采样器进行**空间缩减 + 通道扩展**：

| 过渡 | DWConv 7×7 (stride=2) | Conv 1×1 | 激活 |
|------|----------------------|----------|------|
| Stage 0→1 | [128, 1, 7, 7] g=64 | [128, 128, 1, 1] | GELU |
| Stage 1→2 | [256, 1, 7, 7] g=128 | [256, 256, 1, 1] | GELU |
| Stage 2→3 | [512, 1, 7, 7] g=256 | [512, 512, 1, 1] | GELU |

注意下采样器中的 DWConv 7×7 的 **groups 数等于输入通道数的一半**，使得每个 group 的输出通道为 2，实现通道翻倍。例如 Stage 0→1：weight shape [128, 1, 7, 7] g=64，每组 1 个输入通道产生 2 个输出通道，64→128。

### 3.6 Final Conv + SE 注意力块

Stage 3 输出后经过一个特殊的收尾模块（ONNX 节点 [320-339]）：

```
Stage3 输出 [1, 512, 4, 8]
    │
    ▼ 分组 Conv 3×3 (g=512, 512→1024)     ← 通道翻倍
    │
    ├─→ ReduceMean(H,W) → [1, 1024, 1, 1]
    │       │
    │   Conv 1×1 (1024→64) → ReLU → Conv 1×1 (64→1024) → Sigmoid
    │       │
    │   ◀───┘ SE attention gate
    │
    ▼ element-wise multiply (SE gating)
    │
    ▼ GELU 激活
    │
    ▼ GlobalAveragePool → [1, 1024, 1, 1]
    │
    ▼ Flatten → [1, 1024]
    │
    ▼ Gemm (1024→2048) → [1, 2048]   ← 全连接投影
```

SE（Squeeze-and-Excitation）注意力的压缩比为 16:1（1024→64→1024），在整个 backbone 中**仅出现一次**。

### 3.7 空间分辨率逐级变化表

| 阶段 | 操作 | 输出形状 [N, C, H, W] |
|------|------|----------------------|
| 输入 | Concat(img, big_img) | [1, 24, 128, 256] |
| Stem.0 | Conv 3×3 s2 | [1, 64, 64, 128] |
| Stem.1 | DWConv 3×3 s2 | [1, 64, 32, 64] |
| Stem.2 | Conv 1×1 s1 | [1, 64, 32, 64] |
| Stage 0 | 2× ConvNeXt Block | [1, 64, 32, 64] |
| Down 0→1 | DWConv 7×7 s2 + 1×1 | [1, 128, 16, 32] |
| Stage 1 | 2× ConvNeXt Block | [1, 128, 16, 32] |
| Down 1→2 | DWConv 7×7 s2 + 1×1 | [1, 256, 8, 16] |
| Stage 2 | 6× ConvNeXt Block | [1, 256, 8, 16] |
| Down 2→3 | DWConv 7×7 s2 + 1×1 | [1, 512, 4, 8] |
| Stage 3 | 2× ConvNeXt Block | [1, 512, 4, 8] |
| Final Conv | GConv 3×3 g=512 | [1, 1024, 4, 8] |
| SE + GELU | 通道注意力 | [1, 1024, 4, 8] |
| GAP + FC | 全局池化 + 线性 | [1, 2048] |

总下采样倍率：空间 32× (128→4, 256→8)。

### 3.8 激活函数

**GELU（tanh 近似）**— 在 ONNX 中展开为 6 个算子节点：

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

ONNX 展开:
  Mul(x, x) → x²
  Mul(x, x²) → x³
  Mul(0.044715, x³) → 0.044715x³
  Add(x, 0.044715x³)
  Mul(sqrt(2/π), ...)
  Tanh(...)
  Add(1, tanh_out)
  Mul(x, (1+tanh_out))
  Mul(0.5, ...)
```

整个网络的激活函数分布：

| 激活函数 | 出现次数 | 使用位置 |
|----------|----------|----------|
| GELU (tanh 近似) | 19 个 Tanh 节点 + 128 个 Mul | Stem、Block MLP、下采样器、Final |
| ReLU | 54 个 | 输出头（Summarizer、Hydra） |
| Sigmoid | 1 个 | SE 注意力门控 |

### 3.9 参数分布

| 类别 | 参数量 | 占比 |
|------|--------|------|
| Gemm (全连接) | 16,564,696 | 71.8% |
| Conv (卷积) | 6,490,048 | 28.1% |
| LayerScale/Norm | 3,088 | ~0.01% |
| 其他 (bias等) | 48 | ~0% |
| **总计** | **23,057,880** | **100%** |

全连接层参数占主导（71.8%），主要来自 Backbone 后的 FC 投影层和输出头。Backbone 卷积部分参数效率极高，得益于大量使用 DWConv。

---

## 第4章：输出头架构

### 4.1 GlobalAvgPool → FC 映射

Backbone 输出 [1, 1024, 4, 8] 经全局平均池化和线性投影得到 2048 维特征向量：

```
[1, 1024, 4, 8] → GlobalAvgPool → [1, 1024, 1, 1] → Flatten → [1, 1024]
                → Gemm(1024→2048) → [1, 2048]
```

这个 2048D 向量是所有输出头的**公共根特征**，分三路送入不同的 Summarizer。

### 4.2 三路 Summarizer

Summarizer 的作用是将 2048D 公共特征压缩到 512D，同时通过 ResBlock 增加非线性表达力。三路 Summarizer 结构相同但权重独立：

```
                    ┌─── Summarizer (policy) ────→ Hydra heads (lead/lanes/edges)
                    │
[1, 2048] ─────────├─── Summarizer (no_bottleneck) ──→ Hydra heads (meta/pose/etc.)
                    │
                    └─── Summarizer (hidden_state) ──→ L2 Norm → hidden_state (512D)
```

每个 Summarizer 的内部结构（以 hidden_state 路为例，ONNX 节点 [476-494]）：

```python
class Summarizer:
    def forward(self, x):         # x: [1, 2048]
        x = FC(2048→512)(x)       # [节点476]
        x = relu(x)               # [节点477]
        # ResBlock 1
        h = relu(FC(512→1024)(x)) # [节点478-479]
        h = FC(1024→512)(h)       # [节点480]
        x = relu(x + h)           # [节点481-482]
        # ResBlock 2
        h = relu(FC(512→1024)(x)) # [节点483-484]
        h = FC(1024→512)(h)       # [节点485]
        x = relu(x + h)           # [节点486-487]
        # Final FC + L2 Norm
        x = FC(512→512)(x)        # [节点488]
        x = x / max(||x||₂, ε)   # [节点489-494] L2 归一化
        return x                  # [1, 512]
```

三路 Summarizer 共用同一结构：`FC(2048→512) + ResBlock(512→1024→512)×2`，但末尾不同：
- **policy** 和 **hidden_state**：末尾有 `FC(512→512) + L2 Norm`
- **no_bottleneck**：末尾直接输出 512D（无 L2 Norm）

### 4.3 Hydra 输出头

每个 Summarizer 输出的 512D 向量分别送入**多个并行的 Hydra 头**。每个 Hydra 头结构：

```python
class HydraHead:
    def forward(self, x):             # x: [1, 512] (from Summarizer)
        x = relu(FC(512→hidden)(x))   # in_layer
        # ResBlock ×2
        h = relu(FC(hidden→hidden)(x))
        h = FC(hidden→hidden)(h)
        x = relu(x + h)
        h = relu(FC(hidden→hidden)(x))
        h = FC(hidden→hidden)(h)
        x = relu(x + h)
        return FC(hidden→out)(x)      # final_layer
```

**Policy Summarizer 的 Hydra 头**（节点 [363-413]）：

| 输出 | hidden_dim | out_dim | 备注 |
|------|-----------|---------|------|
| lead | 64 | 144 | 乘以 learned scale |
| lead_prob | 16 | 3 | |
| lane_lines_prob | 16 | 8 | |
| road_edges | 32 | 264 | |
| lane_lines | 64 | 528 | |

注意 lead 输出（144D）在 Hydra 头之后还经过一个 **learned scale 缩放**（ONNX 节点 [413]：`Mul(output, scale[144])`），这是为了稳定 lead 预测的数值范围。

**No_bottleneck Summarizer 的 Hydra 头**（节点 [436-475]）：

| 输出 | hidden_dim | out_dim | 备注 |
|------|-----------|---------|------|
| meta | 64 | 55 | |
| desire_pred | 32 | 32 | |
| pose | 32 | 12 | |
| road_transform | 32 | 12 | |
| wide_from_device_euler | 32 | 6 | |

### 4.4 输出分配与拼接

最终，所有输出在通道维度拼接为 1576D 向量（ONNX 节点 [495] Concat）：

```
Policy Summarizer:
  lead(144) + lead_prob(3) + lane_lines_prob(8) + road_edges(264) + lane_lines(528) = 947D

No_bottleneck Summarizer:
  meta(55) + desire_pred(32) + pose(12) + road_transform(12) + wide_from_device_euler(6) = 117D

Hidden_state Summarizer:
  hidden_state(512D, L2 normalized)

Total: 947 + 117 + 512 = 1576D
```

拼接顺序与 `driving_vision_metadata.pkl` 中的 `output_slices` 一致。

---

## 第5章：输出语义与解析

### 5.1 output_slices 完整映射表

从 `driving_vision_metadata.pkl` 和 `driving_policy_metadata.pkl` 提取的输出切片：

**视觉网络输出 (1576D)**：

| 输出名 | 切片 | 维度 | 语义 |
|--------|------|------|------|
| meta | [0:55] | 55 | 元事件概率 |
| desire_pred | [55:87] | 32 | 意图预测 |
| pose | [87:99] | 12 | 相机位姿 (6DoF × 2) |
| wide_from_device_euler | [99:105] | 6 | 广角对齐角 (3 × 2) |
| road_transform | [105:117] | 12 | 路面变换 (6 × 2) |
| lane_lines | [117:645] | 528 | 车道线 |
| lane_lines_prob | [645:653] | 8 | 车道线概率 |
| road_edges | [653:917] | 264 | 路缘线 |
| lead | [917:1061] | 144 | 前车轨迹 |
| lead_prob | [1061:1064] | 3 | 前车存在概率 |
| hidden_state | [1064:1576] | 512 | 隐藏状态 (送策略网络) |

**策略网络输出 (1000D)**：

| 输出名 | 切片 | 维度 | 语义 |
|--------|------|------|------|
| plan | [0:990] | 990 | 路径规划 (MHP) |
| desire_state | [990:998] | 8 | 意图状态 |
| pad | [-2:] | 2 | 填充 |

### 5.2 MDN 解码

大部分输出使用 **Mixture Density Network (MDN)** 编码，每个预测量包含 (mean, log_std) 对：

```python
# parse_model_outputs.py:44-86
def parse_mdn(self, name, outs, in_N=0, out_N=1, out_shape=None):
    raw = outs[name].reshape((batch, max(in_N, 1), -1))
    n_values = (raw.shape[2] - out_N) // 2
    pred_mu  = raw[:, :, :n_values]                    # 均值
    pred_std = safe_exp(raw[:, :, n_values:2*n_values]) # exp(log_std) → 标准差
```

其中 `safe_exp` 对输入做 clip(-∞, 11) 防止 fp16 溢出。这假设 **Laplace 分布**（因为使用 L1 loss 训练）。

### 5.3 MHP 解码（多假设路径）与当前模型状态

代码支持 **Multi-Hypothesis Prediction (MHP)** ——多个假设 + softmax 权重选择。但通过 `is_mhp()` 运行时检测（`parse_model_outputs.py:88-93`），**当前模型未使用 MHP**：

```python
def is_mhp(self, outs, name, shape):
    if outs[name].shape[1] == 2 * shape:  # 恰好 = 2×shape → 纯 MDN
        return False
    return True                            # 更大 → MHP 编码
```

验证：
- **lead**：144D = 2 × (3×6×4) = 2 × 72 → `is_mhp=False`，纯 MDN
- **plan**：990D = 2 × (1×33×15) = 2 × 495 → `is_mhp=False`，纯 MDN

当使用 MHP 时，结构为 `[in_N 个假设, 每假设含 mean + std + weight]`，并按 softmax 权重排序选择 top-k 假设。当前模型直接输出 `(mean, std)` 对，无需假设选择。

### 5.4 逐输出详解

#### lane_lines（车道线）— 528D

```
4 条车道线 × 33 个距离点 × 2 个几何量 × 2 (mean + std)
= 4 × 33 × 2 × 2 = 528
```

- 4 条车道线：左外、左内、右内、右外
- 33 个距离采样点：`X_IDXS[i] = 192 * (i/32)²`，范围 0~192 米
- 2 个几何量：y（横向偏移）、z（高度）
- 2 个统计量：均值、标准差

#### road_edges（路缘线）— 264D

```
2 条路缘 × 33 个距离点 × 2 个几何量 × 2 (mean + std)
= 2 × 33 × 2 × 2 = 264
```

与车道线结构相同，仅左右各一条。

#### lane_lines_prob（车道线概率）— 8D

8 个值对应 4 条车道线各 2 个概率值（经 sigmoid 解码）。实际使用时取奇数索引：`laneLineProbs = lane_lines_prob[0, 1::2]`（`fill_model_msg.py:107`）。

#### lead（前车轨迹）— 144D

当前模型使用纯 MDN 编码（`is_mhp=False`）：

```
LEAD_MHP_SELECTION=3 × LEAD_TRAJ_LEN=6 × LEAD_WIDTH=4 × 2 (mean + std)
= 3 × 6 × 4 × 2 = 144
```

解码后 reshape 为 `[1, 3, 6, 4]`：
- **3 个 lead 选择**：对应 `LEAD_T_OFFSETS = [0, 2, 4]` 秒的时间偏移
- **6 个时间点**：`LEAD_T_IDXS = [0, 2, 4, 6, 8, 10]` 秒
- **4 个状态量**：距离 x、横向偏移 y、相对速度 v、相对加速度 a

#### lead_prob（前车存在概率）— 3D

3 个 sigmoid 概率，对应三个时间偏移（0s、2s、4s）时前方是否存在领航车辆。

#### meta（元事件概率）— 55D

精细切片定义（`constants.py:74-87`）：

```python
class Meta:
    ENGAGED = slice(0, 1)            # [0]     是否介入 (1D)
    # 未来 2, 4, 6, 8, 10 秒
    GAS_DISENGAGE  = slice(1, 31, 6)  # [1, 7, 13, 19, 25]   (5D)
    BRAKE_DISENGAGE = slice(2, 31, 6) # [2, 8, 14, 20, 26]   (5D)
    STEER_OVERRIDE = slice(3, 31, 6)  # [3, 9, 15, 21, 27]   (5D)
    HARD_BRAKE_3   = slice(4, 31, 6)  # [4, 10, 16, 22, 28]  (5D)
    HARD_BRAKE_4   = slice(5, 31, 6)  # [5, 11, 17, 23, 29]  (5D)
    HARD_BRAKE_5   = slice(6, 31, 6)  # [6, 12, 18, 24, 30]  (5D)
    # 未来 0, 2, 4, 6, 8, 10 秒
    GAS_PRESS    = slice(31, 55, 4)   # [31, 35, 39, 43, 47, 51]  (6D)
    BRAKE_PRESS  = slice(32, 55, 4)   # [32, 36, 40, 44, 48, 52]  (6D)
    LEFT_BLINKER = slice(33, 55, 4)   # [33, 37, 41, 45, 49, 53]  (6D)
    RIGHT_BLINKER = slice(34, 55, 4)  # [34, 38, 42, 46, 50, 54]  (6D)
```

共计：1 (engaged) + 6×5 (脱离事件) + 4×6 (驾驶员操作) = 1 + 30 + 24 = **55D**。所有概率经 sigmoid 解码。

#### desire_pred（意图预测）— 32D

```
4 个未来时间步 × 8 个意图类别 = 32
```

- `DESIRE_PRED_LEN=4`：对应 4 个时间步
- `DESIRE_PRED_WIDTH=8`：8 类意图（直行、左变道、右变道等）
- 经 softmax 解码（每个时间步独立 softmax）

#### pose（相机位姿）— 12D

```
6 个自由度 × 2 (mean + std) = 12
POSE_WIDTH = 6: [trans_x, trans_y, trans_z, rot_roll, rot_pitch, rot_yaw]
```

用于相机里程计（`cameraOdometry` 消息），输出包括平移分量（trans）和旋转分量（rot）及其标准差。

#### road_transform（路面变换）— 12D

```
6 个参数 × 2 (mean + std) = 12
```

描述路面几何/姿态的变换参数。实际使用中仅取前 3 维（平移分量）作为 `roadTransformTrans`。

#### wide_from_device_euler（广角对齐角）— 6D

```
3 个欧拉角 × 2 (mean + std) = 6
```

设备坐标系到广角相机的旋转角（roll, pitch, yaw），用于多相机几何对齐。

#### hidden_state（隐藏状态）— 512D

经 L2 归一化的 512 维特征向量，直接送入策略网络作为视觉特征表示。这是视觉网络与策略网络之间的**唯一信息瓶颈**。

---

## 第6章：策略网络

### 6.1 输入

| 输入 | 形状 | 类型 | 说明 |
|------|------|------|------|
| features_buffer | [1, 25, 512] | fp32 | 视觉特征时序缓冲 |
| desire_pulse | [1, 25, 8] | fp32 | 驾驶意图脉冲 |
| traffic_convention | [1, 2] | fp32 | 交通惯例 (左驾/右驾) |

`features_buffer` 包含最近 25 帧（= 5 秒 @ 5Hz）的 hidden_state。`desire_pulse` 以相同的 25 帧窗口记录意图脉冲信号。

### 6.2 Encoder（编码器）

策略网络的编码器将三种输入分别编码后融合：

**features encoder（ONNX 节点 [0-9]）**：

```python
# 取最新一帧特征
feat = features_buffer[:, -1, :]            # [1, 512]  [节点0-1: Gather]
h = FC(512→512)(feat)                        # [节点4]
h = swish(h)                                 # Sigmoid * h [节点5-7]
feat_encoded = FC(512→512)(h)                # [节点8-9] → [1, 512]
```

**desire encoder（ONNX 节点 [3, 10-15]）**：

```python
desire_flat = desire_pulse.reshape(1, 200)   # [1, 25×8=200] [节点3]
h = FC(200→512)(desire_flat)                  # [节点10]
h = swish(h)                                  # [节点11-12]
desire_encoded = FC(512→512)(h)               # [节点13]
desire_encoded = desire_encoded.unsqueeze(1)  # [1, 1, 512] [节点15]
```

**traffic_convention encoder（ONNX 节点 [16-21]）**：

```python
h = FC(2→512)(traffic_convention)             # [节点16]
h = swish(h)                                  # [节点17-18]
tc_encoded = FC(512→512)(h)                    # [节点19]
tc_encoded = tc_encoded.unsqueeze(1)           # [1, 1, 512] [节点21]
```

所有 encoder 使用 **Swish 激活**（`x * sigmoid(x)`），而非 backbone 中的 GELU。

### 6.3 序列构建与位置编码

编码后的特征融合为一个 9 步序列：

```python
# 9 步序列构建（ONNX 节点 [22-24]）：
# feat_encoded 通过 MatMul 映射为 [1, 9, 512]（类似线性投影到 9 个位置）
# 实际上：feat_encoded + positional_embedding[1, 9, 512]
sequence = feat_encoded + pos_embed           # [1, 9, 512]  [节点22]
sequence = sequence + desire_encoded          # 广播加 [节点23]
sequence = sequence + tc_encoded              # 广播加 [节点24]
```

位置编码（`onnx::Add_207[1, 9, 512]`）是**可学习的固定嵌入**。

### 6.4 GPT 风格 Transformer（1 层）

策略网络使用**单层因果 Transformer**（ONNX 节点 [25-72]）：

```python
# Pre-LayerNorm
x_norm = LayerNorm(sequence)                   # [节点25]

# Multi-Head Attention
qkv = FC(512→1536)(x_norm)                     # [节点26-27] 512→1536 = 3×512
qkv = qkv.reshape(1, 9, 3, n_heads, head_dim)  # [节点30]
q, k, v = split(qkv.transpose(2,0,3,1,4))      # [节点31-39]

# Scaled dot-product attention with causal mask
attn = (q @ k.T) * scale                        # [节点40-43]
attn = where(causal_mask, -inf, attn)           # [节点44-45] 因果掩码
attn = softmax(attn)                            # [节点46]
out = attn @ v                                  # [节点47]

# Output projection
out = out.transpose(0,2,1,3).reshape(1, 9, 512) # [节点48-50]
out = FC(512→512)(out)                           # [节点51-52]
x = sequence + out                               # [节点53] 残差连接

# FFN with Pre-LayerNorm
x_norm = LayerNorm(x)                            # [节点54]
h = FC(512→2048)(x_norm)                          # [节点55-56]
h = gelu(h)                                       # [节点57-69] GELU tanh 近似
h = FC(2048→512)(h)                               # [节点70-71]
x = x + h                                         # [节点72] 残差连接
```

因果掩码 `onnx::Where_216[1, 1, 9, 9]` 确保每个位置只能注意到自身及之前的位置。FFN 扩展比为 4× (512→2048→512)。

### 6.5 输出头

从 Transformer 输出取**最后一个时间步**（Gather index=-1，节点 [73]），送入两个独立的 Hydra 头：

**plan 头**（ONNX 节点 [74-98, 100]）：

```python
# Summarizer: FC(512→1024)→ReLU→FC(1024→512) ResBlock×2
x = transformer_out[:, -1, :]          # [1, 512]
x = summarizer(x)                       # [1, 512] → ResBlock×2 in 512D
# Hydra head: FC(512→256), ResBlock×2 in 256D
x = hydra_in(x)                         # [1, 256]
x = resblocks(x)                        # ResBlock×2 in 256D
plan_raw = FC(256→990)(x)               # [节点98] → [1, 990]
plan = plan_raw * learned_scale[990]     # [节点100] 缩放
```

plan 的 990D 结构（当前为纯 MDN，`is_mhp=False`）：

```
PLAN_MHP_SELECTION=1 × IDX_N=33 × PLAN_WIDTH=15 × 2 (mean + std)
= 1 × 33 × 15 × 2 = 990
```

解码后 reshape 为 `[1, 33, 15]`，33 个时间点上的 15 个状态量：

```python
class Plan:                                # constants.py:67-72
    POSITION = slice(0, 3)                 # xyz 位置
    VELOCITY = slice(3, 6)                 # xyz 速度
    ACCELERATION = slice(6, 9)             # xyz 加速度
    T_FROM_CURRENT_EULER = slice(9, 12)    # 相对当前的欧拉角
    ORIENTATION_RATE = slice(12, 15)       # 角速率
```

33 个时间采样点对应 `T_IDXS[i] = 10 * (i/32)²`，范围 0~10 秒。

**desire_state 头**（ONNX 节点 [86-99]）：

```python
x = hydra_in(x)                         # FC(512→32)
x = resblocks(x)                        # ResBlock×2 in 32D
desire_state = FC(32→8)(x)              # [节点99] → [1, 8]
```

经 softmax 解码为 8 类意图概率分布。

### 6.6 InputQueues 时序缓冲

`InputQueues`（`modeld.py:123-196`）管理策略网络的时序输入：

| 参数 | 值 | 说明 |
|------|----|------|
| model_fps | 5 Hz | 策略网络的上下文频率 |
| env_fps | 20 Hz | 视觉网络运行频率 |
| n_frames_input | 2 | 视觉网络的帧数 |
| 窗口长度 | 25 帧 | 5秒 @ 5Hz |

核心机制：
- **features_buffer**：20Hz hidden_state 到达时，按 4:1 采样率取最新 25 帧，形成 [1, 25, 512]
- **desire_pulse**：在每个 4 帧区间内取 **max-pooling**（只要有脉冲即为 1），避免短暂脉冲丢失
- **滑动窗口**：每次新数据入队时，窗口左移腾出空间（`modeld.py:170-172`）

desire 处理的特殊逻辑：只在**上升沿**产生脉冲（`modeld.py:265-267`），避免持续输入导致模型内部状态错误累积：

```python
new_desire = np.where(inputs['desire_pulse'] - self.prev_desire > .99,
                      inputs['desire_pulse'], 0)
```

---

## 第7章：端到端推理流程

### 7.1 完整数据流图

```
┌────────────────────── modeld 主循环 (modeld.py:388+) ──────────────────────┐
│                                                                              │
│  1. VisionIPC 接收帧                                                         │
│     vipc_client_main.recv() → buf_main (窄角/主路)                          │
│     vipc_client_extra.recv() → buf_extra (广角)                             │
│     帧同步：|timestamp_sof 差| < 10ms                                       │
│                                                                              │
│  2. 计算 Warp Matrix (每次 liveCalibration 更新时)                          │
│     device_from_calib_euler → get_warp_matrix() → model_transform           │
│                                                                              │
│  3. OpenCL 预处理 (commonmodel.cc)                                           │
│     warpPerspective: NV12 → 512×256 Y/U/V                                  │
│     loadyuv: Y(4ch) + U(1ch) + V(1ch) = 6ch                               │
│     temporal: 旧帧(6ch) + 新帧(6ch) = 12ch                                 │
│     img[1,12,128,256] + big_img[1,12,128,256] → uint8                      │
│                                                                              │
│  4. 视觉网络推理 (tinygrad)                                                  │
│     vision_run(**vision_inputs) → [1, 1576] fp16/fp32                       │
│     slice_outputs → 命名子向量                                               │
│     parse_vision_outputs → MDN/sigmoid/softmax 解码                         │
│                                                                              │
│  5. 策略网络推理 (tinygrad)                                                  │
│     features_buffer ← hidden_state 入队                                      │
│     policy_run(**policy_inputs) → [1, 1000] fp16/fp32                       │
│     parse_policy_outputs → MHP/softmax 解码                                 │
│                                                                              │
│  6. 后处理与发布                                                              │
│     get_action_from_model → desired_curvature, desired_acceleration          │
│     fill_model_msg → modelV2 (扩展) + drivingModelData (核心)               │
│     fill_pose_msg → cameraOdometry                                          │
│     pm.send('modelV2'), pm.send('drivingModelData'), pm.send('cameraOdom')  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 20Hz 推理时序

模型以 **20 Hz** 运行（50ms 周期），时序预算：

```
|← ─ ─ ─ ─ ─ ─ 50ms ─ ─ ─ ─ ─ ─ →|
├─────┬──────────────┬──────────────┤
│recv │  model.run() │ post+publish │
│~2ms │   ~35-45ms   │    ~3ms      │
├─────┴──────────────┴──────────────┤
│        其中 model.run():          │
│  OpenCL prep: ~5ms                │
│  Vision forward: ~20ms            │
│  Policy forward: ~10ms            │
│  Output parse: ~2ms               │
└───────────────────────────────────┘
```

当发生**丢帧**时（`vipc_dropped_frames > 0`），仅执行 `prepare()` 而跳过前向推理（`prepare_only=True`），保持输入缓冲同步但不产生输出。

### 7.3 硬件后端

| 平台 | DEV 环境变量 | 后端 | 说明 |
|------|-------------|------|------|
| comma 3X (TICI) | QCOM | Qualcomm OpenCL | 默认，零拷贝优化 |
| NVIDIA GPU | CUDA | CUDA | 开发/测试 |
| AMD USB GPU | AMD | AMD OpenCL | `USBGPU=1` 激活 |
| x86 开发机 | CPU | CPU | 回退方案 |

在 TICI 上，通过 `qcom_tensor_from_opencl_address()` 实现 OpenCL 显存到 tinygrad 张量的**零拷贝映射**，避免 GPU↔CPU 数据传输：

```python
if TICI and not USBGPU:
    # OpenCL 显存直接映射为 tinygrad 张量，仅首次创建
    if key not in self.vision_inputs:
        self.vision_inputs[key] = qcom_tensor_from_opencl_address(
            imgs_cl[key].mem_address, shape, dtype=dtypes.uint8)
```

### 7.4 tinygrad pkl 格式

模型以 Python pickle 格式存储预编译的 tinygrad 计算图：

```
selfdrive/modeld/models/
├── driving_vision_tinygrad.pkl      # QCOM/通用 (52MB)
├── driving_vision_tinygrad_cuda.pkl # CUDA 特化 (49MB)
├── driving_policy_tinygrad.pkl      # QCOM/通用 (15MB)
└── driving_policy_tinygrad_cuda.pkl # CUDA 特化 (14MB)
```

加载后直接调用：`vision_run(**inputs)` 返回 tinygrad Tensor，通过 `.contiguous().realize().uop.base.buffer.numpy()` 取回 numpy 数组。

---

## 第8章：架构设计特征总结

### 8.1 与标准 ConvNeXt 的差异

| 特征 | 标准 ConvNeXt V1 | openpilot 变体 |
|------|-----------------|----------------|
| Token Mixer | 7×7 DWConv | **3×3 DWConv + 7×7 DWConv** (双尺度串行) |
| Normalization | LayerNorm (after DWConv) | **无 Normalization** |
| 残差范围 | 包裹整个 block | **仅包裹 MLP 部分** (不含 token_mixer) |
| 下采样器 | 独立的 LN + Conv 2×2 s2 | **DWConv 7×7 s2 + Conv 1×1 + GELU** |
| SE 注意力 | 无 | **Final Conv 后单次 SE** |
| 输入归一化 | BatchNorm/InstanceNorm | **逐通道 Mean/Std** (无可学习参数) |
| Stem | Patchify (Conv 4×4 s4) | **3 层渐进: 3×3 s2 + DW3×3 s2 + 1×1** |
| 激活 | GELU | GELU (backbone) + **ReLU** (heads) + **Swish** (policy encoder) |

核心设计思路：**去除所有归一化层**（无 BN/LN），用 LayerScale 替代稳定训练。这减少了推理时的计算开销，对嵌入式部署友好。

### 8.2 多任务输出设计

采用 **Summarizer + Hydra** 的两级输出架构：

```
Backbone → FC(2048D)
    ├── Summarizer A (512D) → Hydra: lead, lanes, edges     (感知类)
    ├── Summarizer B (512D) → Hydra: meta, pose, etc.       (姿态类)
    └── Summarizer C (512D) → L2 Norm → hidden_state        (时序传递)
```

这种设计的优势：
1. **任务解耦**：不同类型的输出有独立的特征压缩路径，减少任务间干扰
2. **灵活扩展**：添加新输出只需在对应 Summarizer 下增加 Hydra 头
3. **信息瓶颈**：hidden_state 经 L2 归一化后传递给策略网络，防止信息过载

### 8.3 时序处理策略

openpilot 在两个层次处理时序信息：

| 层次 | 方法 | 时间跨度 | 机制 |
|------|------|----------|------|
| 视觉网络 | **帧拼接** | 200ms (2帧) | 通道维 concat |
| 策略网络 | **Transformer** | 5s (25帧) | 因果自注意力 |

视觉网络的帧拼接仅提供**短时运动线索**（光流等效信息），长期时序依赖完全由策略网络的 Transformer 处理。这种分层设计平衡了计算效率和时序建模能力。

### 8.4 效率设计特征

1. **fp16 推理**：ONNX 模型以 fp16 存储权重，减半内存占用和带宽需求
2. **无 Normalization 层**：省去 BN/LN 的统计计算和额外内存访问
3. **大量 DWConv**：60 个卷积操作中 31 个为 DWConv，参数效率极高
4. **LayerScale**：仅用一个可学习标量向量（per-channel）替代复杂的归一化层
5. **零拷贝管线**：TICI 上 OpenCL→tinygrad 的显存直接映射
6. **单层 Transformer**：策略网络仅 1 层 Transformer，序列长度仅 9

---

## 附录A：源码文件索引表

| 文件路径 | 职责 |
|----------|------|
| `selfdrive/modeld/modeld.py` | 主循环：帧接收、模型推理、消息发布 |
| `selfdrive/modeld/constants.py` | 模型常量定义（维度、索引、切片） |
| `selfdrive/modeld/parse_model_outputs.py` | 输出解析（MDN/MHP/sigmoid/softmax） |
| `selfdrive/modeld/fill_model_msg.py` | cereal 消息填充与发布 |
| `selfdrive/modeld/models/commonmodel.h` | 模型帧处理 C++ 头文件 |
| `selfdrive/modeld/models/commonmodel.cc` | DrivingModelFrame 实现 |
| `selfdrive/modeld/models/commonmodel_pyx.pyx` | Cython 绑定层 |
| `selfdrive/modeld/transforms/transform.cl` | OpenCL 透视变换 kernel |
| `selfdrive/modeld/transforms/transform.h` | 变换接口定义 |
| `selfdrive/modeld/transforms/loadyuv.cl` | OpenCL YUV 重组 kernel |
| `selfdrive/modeld/transforms/loadyuv.h` | YUV 加载接口定义 |
| `common/transformations/model.py` | 虚拟相机参数、warp_matrix 计算 |
| `common/transformations/camera.py` | 相机配置、坐标系变换 |
| `selfdrive/modeld/runners/tinygrad_helpers.py` | tinygrad 推理辅助 |
| `selfdrive/modeld/models/README.md` | 官方模型输入格式说明 |

## 附录B：算子类型统计表

### 视觉网络 (driving_vision.onnx) — 496 个节点，19 种算子

| 算子 | 数量 | 占比 | 主要用途 |
|------|------|------|----------|
| Mul | 128 | 25.8% | GELU 展开、LayerScale、SE gate |
| Constant | 78 | 15.7% | GELU 常数、shape 常数 |
| Add | 70 | 14.1% | 残差连接、GELU 展开、bias |
| Gemm | 66 | 13.3% | Summarizer、Hydra 全连接 |
| Conv | 60 | 12.1% | Backbone 卷积 |
| Relu | 54 | 10.9% | 输出头激活 |
| Tanh | 19 | 3.8% | GELU tanh 近似 |
| Div | 3 | 0.6% | 输入归一化、L2 Norm |
| Cast | 2 | 0.4% | uint8→fp16 |
| Concat | 2 | 0.4% | 输入拼接、输出拼接 |
| Flatten | 2 | 0.4% | GAP 后展平 |
| ReduceL2 | 2 | 0.4% | L2 归一化 |
| Clip | 2 | 0.4% | L2 Norm ε 裁剪 |
| Shape | 2 | 0.4% | L2 Norm broadcast |
| Expand | 2 | 0.4% | L2 Norm broadcast |
| Sub | 1 | 0.2% | 减均值 |
| ReduceMean | 1 | 0.2% | SE squeeze |
| Sigmoid | 1 | 0.2% | SE gate |
| GlobalAveragePool | 1 | 0.2% | 全局池化 |

### 策略网络 (driving_policy.onnx) — 102 个节点，18 种算子

| 算子 | 数量 | 主要用途 |
|------|------|----------|
| Constant | 17 | 形状常数、掩码 |
| Add | 17 | 残差、位置编码、bias |
| Gemm | 16 | Encoder FC、Hydra 全连接 |
| Mul | 11 | GELU、Swish、scale |
| Relu | 10 | 输出头激活 |
| MatMul | 8 | Attention QKV、FFN |
| Reshape | 3 | 注意力头变形 |
| Sigmoid | 3 | Swish 激活 |
| Transpose | 3 | 注意力计算 |
| Squeeze | 3 | 移除维度 |
| Gather | 2 | 索引取帧 |
| Unsqueeze | 2 | 增加维度 |
| LayerNormalization | 2 | Pre-LN Transformer |
| Split | 1 | QKV 拆分 |
| Where | 1 | 因果掩码 |
| Softmax | 1 | 注意力权重 |
| Tanh | 1 | GELU (FFN) |
| Concat | 1 | 输出拼接 |

## 附录C：完整卷积操作清单

视觉网络共 **60 个 Conv 操作**（29 个标准 Conv + 31 个 DWConv）：

| # | 位置 | 类型 | Kernel | Stride | Groups | 权重形状 | 所属模块 |
|---|------|------|--------|--------|--------|----------|----------|
| 0 | Stem | Conv | 3×3 | 2 | 1 | [64, 24, 3, 3] | stem.0 |
| 1 | Stem | DWConv | 3×3 | 2 | 64 | [64, 1, 3, 3] | stem.1 |
| 2 | Stem | Conv | 1×1 | 1 | 1 | [64, 64, 1, 1] | stem.2 |
| 3 | S0.B0 | DWConv | 3×3 | 1 | 64 | [64, 1, 3, 3] | token_mixer |
| 4 | S0.B0 | DWConv | 7×7 | 1 | 64 | [64, 1, 7, 7] | mlp.conv |
| 5 | S0.B0 | Conv | 1×1 | 1 | 1 | [192, 64, 1, 1] | mlp.fc1 |
| 6 | S0.B0 | Conv | 1×1 | 1 | 1 | [64, 192, 1, 1] | mlp.fc2 |
| 7 | S0.B1 | DWConv | 3×3 | 1 | 64 | [64, 1, 3, 3] | token_mixer |
| 8 | S0.B1 | DWConv | 7×7 | 1 | 64 | [64, 1, 7, 7] | mlp.conv |
| 9 | S0.B1 | Conv | 1×1 | 1 | 1 | [192, 64, 1, 1] | mlp.fc1 |
| 10 | S0.B1 | Conv | 1×1 | 1 | 1 | [64, 192, 1, 1] | mlp.fc2 |
| 11 | Down0→1 | DWConv | 7×7 | 2 | 64 | [128, 1, 7, 7] | downsample |
| 12 | Down0→1 | Conv | 1×1 | 1 | 1 | [128, 128, 1, 1] | projection |
| 13 | S1.B0 | DWConv | 3×3 | 1 | 128 | [128, 1, 3, 3] | token_mixer |
| 14 | S1.B0 | DWConv | 7×7 | 1 | 128 | [128, 1, 7, 7] | mlp.conv |
| 15 | S1.B0 | Conv | 1×1 | 1 | 1 | [384, 128, 1, 1] | mlp.fc1 |
| 16 | S1.B0 | Conv | 1×1 | 1 | 1 | [128, 384, 1, 1] | mlp.fc2 |
| 17 | S1.B1 | DWConv | 3×3 | 1 | 128 | [128, 1, 3, 3] | token_mixer |
| 18 | S1.B1 | DWConv | 7×7 | 1 | 128 | [128, 1, 7, 7] | mlp.conv |
| 19 | S1.B1 | Conv | 1×1 | 1 | 1 | [384, 128, 1, 1] | mlp.fc1 |
| 20 | S1.B1 | Conv | 1×1 | 1 | 1 | [128, 384, 1, 1] | mlp.fc2 |
| 21 | Down1→2 | DWConv | 7×7 | 2 | 128 | [256, 1, 7, 7] | downsample |
| 22 | Down1→2 | Conv | 1×1 | 1 | 1 | [256, 256, 1, 1] | projection |
| 23 | S2.B0 | DWConv | 3×3 | 1 | 256 | [256, 1, 3, 3] | token_mixer |
| 24 | S2.B0 | DWConv | 7×7 | 1 | 256 | [256, 1, 7, 7] | mlp.conv |
| 25 | S2.B0 | Conv | 1×1 | 1 | 1 | [768, 256, 1, 1] | mlp.fc1 |
| 26 | S2.B0 | Conv | 1×1 | 1 | 1 | [256, 768, 1, 1] | mlp.fc2 |
| 27 | S2.B1 | DWConv | 3×3 | 1 | 256 | [256, 1, 3, 3] | token_mixer |
| 28 | S2.B1 | DWConv | 7×7 | 1 | 256 | [256, 1, 7, 7] | mlp.conv |
| 29 | S2.B1 | Conv | 1×1 | 1 | 1 | [768, 256, 1, 1] | mlp.fc1 |
| 30 | S2.B1 | Conv | 1×1 | 1 | 1 | [256, 768, 1, 1] | mlp.fc2 |
| 31 | S2.B2 | DWConv | 3×3 | 1 | 256 | [256, 1, 3, 3] | token_mixer |
| 32 | S2.B2 | DWConv | 7×7 | 1 | 256 | [256, 1, 7, 7] | mlp.conv |
| 33 | S2.B2 | Conv | 1×1 | 1 | 1 | [768, 256, 1, 1] | mlp.fc1 |
| 34 | S2.B2 | Conv | 1×1 | 1 | 1 | [256, 768, 1, 1] | mlp.fc2 |
| 35 | S2.B3 | DWConv | 3×3 | 1 | 256 | [256, 1, 3, 3] | token_mixer |
| 36 | S2.B3 | DWConv | 7×7 | 1 | 256 | [256, 1, 7, 7] | mlp.conv |
| 37 | S2.B3 | Conv | 1×1 | 1 | 1 | [768, 256, 1, 1] | mlp.fc1 |
| 38 | S2.B3 | Conv | 1×1 | 1 | 1 | [256, 768, 1, 1] | mlp.fc2 |
| 39 | S2.B4 | DWConv | 3×3 | 1 | 256 | [256, 1, 3, 3] | token_mixer |
| 40 | S2.B4 | DWConv | 7×7 | 1 | 256 | [256, 1, 7, 7] | mlp.conv |
| 41 | S2.B4 | Conv | 1×1 | 1 | 1 | [768, 256, 1, 1] | mlp.fc1 |
| 42 | S2.B4 | Conv | 1×1 | 1 | 1 | [256, 768, 1, 1] | mlp.fc2 |
| 43 | S2.B5 | DWConv | 3×3 | 1 | 256 | [256, 1, 3, 3] | token_mixer |
| 44 | S2.B5 | DWConv | 7×7 | 1 | 256 | [256, 1, 7, 7] | mlp.conv |
| 45 | S2.B5 | Conv | 1×1 | 1 | 1 | [768, 256, 1, 1] | mlp.fc1 |
| 46 | S2.B5 | Conv | 1×1 | 1 | 1 | [256, 768, 1, 1] | mlp.fc2 |
| 47 | Down2→3 | DWConv | 7×7 | 2 | 256 | [512, 1, 7, 7] | downsample |
| 48 | Down2→3 | Conv | 1×1 | 1 | 1 | [512, 512, 1, 1] | projection |
| 49 | S3.B0 | DWConv | 3×3 | 1 | 512 | [512, 1, 3, 3] | token_mixer |
| 50 | S3.B0 | DWConv | 7×7 | 1 | 512 | [512, 1, 7, 7] | mlp.conv |
| 51 | S3.B0 | Conv | 1×1 | 1 | 1 | [1536, 512, 1, 1] | mlp.fc1 |
| 52 | S3.B0 | Conv | 1×1 | 1 | 1 | [512, 1536, 1, 1] | mlp.fc2 |
| 53 | S3.B1 | DWConv | 3×3 | 1 | 512 | [512, 1, 3, 3] | token_mixer |
| 54 | S3.B1 | DWConv | 7×7 | 1 | 512 | [512, 1, 7, 7] | mlp.conv |
| 55 | S3.B1 | Conv | 1×1 | 1 | 1 | [1536, 512, 1, 1] | mlp.fc1 |
| 56 | S3.B1 | Conv | 1×1 | 1 | 1 | [512, 1536, 1, 1] | mlp.fc2 |
| 57 | Final | GConv | 3×3 | 1 | 512 | [1024, 1, 3, 3] | channel expand |
| 58 | SE | Conv | 1×1 | 1 | 1 | [64, 1024, 1, 1] | SE squeeze |
| 59 | SE | Conv | 1×1 | 1 | 1 | [1024, 64, 1, 1] | SE excite |

> 注：S=Stage, B=Block, Down=下采样器。GConv 表示分组卷积（group=512, 输出通道翻倍）。
