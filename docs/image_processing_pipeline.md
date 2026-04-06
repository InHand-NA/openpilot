# openpilot 图像处理流水线：从原始采集到模型输入

本文档详细介绍 openpilot 中图像数据从相机传感器采集到送入 driving vision 模型的完整处理过程。Road Camera（窄视角）和 Wide Camera（广视角）分开介绍。

## 目录

1. [总体架构](#1-总体架构)
2. [硬件平台与相机配置](#2-硬件平台与相机配置)
3. [阶段一：传感器采集与 ISP 处理（camerad）](#3-阶段一传感器采集与-isp-处理camerad)
4. [阶段二：VisionIPC 传输](#4-阶段二visionipc-传输)
5. [阶段三：modeld 接收与帧同步](#5-阶段三modeld-接收与帧同步)
6. [阶段四：标定变换矩阵计算](#6-阶段四标定变换矩阵计算)
7. [阶段五：OpenCL 透视变换](#7-阶段五opencl-透视变换)
8. [阶段六：YUV 通道重排（loadyuv）](#8-阶段六yuv-通道重排loadyuv)
9. [阶段七：时序帧组装与最终张量](#9-阶段七时序帧组装与最终张量)
10. [Road Camera 与 Wide Camera 对比](#10-road-camera-与-wide-camera-对比)
11. [关键参数汇总](#11-关键参数汇总)
12. [附录：坐标系定义](#12-附录坐标系定义)

---

## 1. 总体架构

完整数据流如下：

```
传感器 (MIPI CSI-2, RAW12 Bayer)
    |
    v
camerad / ISP (Spectra IFE)
    |  黑电平校正 → 线性化 → 暗角校正(仅road) → 去拜耳 → 白平衡
    |  → 色彩校正 → 伽马校正 → RGB→YUV(BT.601) → 下采样
    |
    v
YUV NV12 (1928x1208 或 1344x760)
    |
    v  VisionIPC 共享内存（18 缓冲环形队列）
    |
modeld
    |
    ├─ 1. 接收帧 + 时间戳同步
    |
    ├─ 2. 计算 3x3 warp 变换矩阵（基于实时标定 rpyCalib + 相机内参）
    |
    ├─ 3. OpenCL warpPerspective 透视变换
    |     输入: NV12 1928x1208 → 输出: Y 512x256, U 256x128, V 256x128
    |
    ├─ 4. loadyuv 通道重排
    |     Y 棋盘拆分为 4 个子通道 + U + V = 6 通道 (6, 128, 256)
    |
    ├─ 5. 时序帧组装
    |     当前帧 + 历史帧 → (1, 12, 128, 256) uint8
    |
    v
driving_vision 模型输入
    img:     (1, 12, 128, 256) uint8  ← Road Camera
    big_img: (1, 12, 128, 256) uint8  ← Wide Camera
```

**关键源文件索引**：

| 功能模块 | 文件路径 |
|---------|---------|
| 相机配置 | `system/camerad/cameras/hw.h` |
| 传感器驱动 | `system/camerad/sensors/ox03c10.cc`, `os04c10.cc` |
| ISP 处理 | `system/camerad/cameras/spectra.cc`, `cameras/ife.h` |
| VisionIPC 发送 | `system/camerad/cameras/camera_common.cc` |
| 模型主循环 | `selfdrive/modeld/modeld.py` |
| 变换矩阵计算 | `common/transformations/model.py`, `camera.py` |
| OpenCL 透视变换 | `selfdrive/modeld/transforms/transform.cc`, `transform.cl` |
| YUV 重排 | `selfdrive/modeld/transforms/loadyuv.cc`, `loadyuv.cl` |
| C++ 帧处理 | `selfdrive/modeld/models/commonmodel.cc`, `commonmodel.h` |
| 模型常量 | `selfdrive/modeld/constants.py` |
| 相机内参 | `common/transformations/camera.py` |
| 模型内参 | `common/transformations/model.py` |

---

## 2. 硬件平台与相机配置

### 2.1 comma 3X 三相机平台

comma 3X（代号 TICI）搭载三个相机，本文只关注前向两个：

| 属性 | Road Camera (窄视角) | Wide Camera (广视角) |
|------|---------------------|---------------------|
| 相机编号 | 1 | 0 |
| 物理焦距 | 8.0 mm | 1.71 mm |
| VisionIPC 流类型 | `VISION_STREAM_ROAD` | `VISION_STREAM_WIDE_ROAD` |
| cereal 消息名 | `roadCameraState` | `wideRoadCameraState` |
| 暗角校正 | 启用 | 不启用 |
| ISP 输出类型 | IFE 处理后 | IFE 处理后 |

> 源码：`system/camerad/cameras/hw.h:32-54`

### 2.2 传感器参数

**OX03C10 传感器**（Road Camera 和 Wide Camera 共用同款传感器）：

| 参数 | 值 |
|------|-----|
| 原始分辨率 | 1928 x 1208 |
| 像素大小 | 3.0 μm |
| 比特深度 | 12-bit (RAW12) |
| 帧率 | ~20 FPS (模型运行频率) |
| Bayer 模式 | GRGRGR |
| 黑电平 | 0 |

> 源码：`system/camerad/sensors/ox03c10.cc:27-55`

### 2.3 软件中的相机内参

openpilot 在软件层面为每个相机定义了等效的像素焦距（focal length in pixels），用于图像变换计算：

```python
# common/transformations/camera.py:49-53
_ar_ox_fisheye = CameraConfig(1928, 1208, 567.0)     # Wide Camera: f=567 像素
_ar_ox_config = DeviceCameraConfig(
    fcam=CameraConfig(1928, 1208, 2648.0),            # Road Camera: f=2648 像素
    dcam=_ar_ox_fisheye,
    ecam=_ar_ox_fisheye                                # Wide Camera: f=567 像素
)
```

**各相机的内参矩阵 K**（由 `CameraConfig.intrinsics` 属性自动生成）：

Road Camera (`fcam`):
```
K_road = [[2648.0,    0.0,  964.0],
           [   0.0, 2648.0,  604.0],
           [   0.0,    0.0,    1.0]]
```

Wide Camera (`ecam`):
```
K_wide = [[567.0,   0.0, 964.0],
           [  0.0, 567.0, 604.0],
           [  0.0,   0.0,   1.0]]
```

> 注意：`cx = width/2 = 1928/2 = 964`，`cy = height/2 = 1208/2 = 604`。
> 源码：`common/transformations/camera.py:19-25`

---

## 3. 阶段一：传感器采集与 ISP 处理（camerad）

### 3.1 ISP 处理流程

传感器输出 RAW12 Bayer 数据后，由 Qualcomm Spectra ISP 的 IFE（Image Front End）单元进行一系列硬件加速处理：

```
RAW12 Bayer
    |
    v
1. 黑电平校正 (Black Level Correction)
    |  OX03C10: black_level = 0
    |  减去传感器暗电流偏移
    v
2. 线性化 (Linearization)
    |  使用 36 元素查找表 (LUT)
    |  修正传感器非线性响应
    v
3. 暗角校正 (Vignetting Correction) —— 仅 Road Camera
    |  使用 221 元素 LUT (GRR + GBB)
    |  补偿镜头边缘光衰减
    v
4. 去拜耳 (Demosaicing / Debayer)
    |  Bayer GRGR 阵列 → 全彩 RGB
    v
5. 白平衡 (White Balance)
    |  R/G/B 通道增益调整
    v
6. 色彩校正 (Color Correction)
    |  3x3 色彩校正矩阵 (CCM)
    v
7. 伽马校正 (Gamma Correction)
    |  使用 64 元素 RGB 独立 LUT
    |  OX03C10: fx = -0.507*exp(-12.54*x) + 0.966*x^0.5 - 0.473*x + 0.507
    v
8. RGB → YUV 转换 (BT.601)
    |  Y = 0.299R + 0.587G + 0.114B
    |  U = -0.169R - 0.331G + 0.500B + 128
    |  V = 0.500R - 0.419G - 0.081B + 128
    v
9. 色度下采样 → NV12 格式
    Y: 1928x1208 (全分辨率)
    UV: 964x604 (半分辨率, U/V 交错存储)
```

> 关键源码：
> - ISP 寄存器配置: `system/camerad/cameras/ife.h:108-234`
> - IFE 初始化: `system/camerad/cameras/spectra.cc:702-898`
> - 传感器特定参数: `system/camerad/sensors/ox03c10.cc:70-107`

### 3.2 NV12 输出格式

ISP 输出的 NV12 格式在内存中的布局：

```
偏移 0:           Y 平面 (stride × height 字节)
                  每行 stride 字节，共 height 行
                  每像素 1 字节 (0-255)

偏移 uv_offset:   UV 平面 (stride × height/2 字节)
                  U 和 V 交错排列: [U0, V0, U1, V1, ...]
                  每行 stride 字节，共 height/2 行
```

UV 平面中 U 和 V 交错存储（NV12 格式的特征），这一点在后续 modeld 的变换处理中很重要。

### 3.3 自动曝光（AE）

camerad 同时运行自动曝光算法，通过分析 Y 平面亮度直方图来调整传感器的曝光时间和模拟增益。这不直接影响模型输入的处理流程，但影响图像的亮度和对比度。

> 源码：`system/camerad/cameras/camera_qcom2.cc:126-222`

---

## 4. 阶段二：VisionIPC 传输

camerad 通过 VisionIPC 共享内存机制将 NV12 帧发送给下游消费者（modeld 等）。

### 4.1 发送机制

```cpp
// system/camerad/cameras/camera_common.cc:45-61
void CameraBuf::sendFrameToVipc() {
    cur_yuv_buf = vipc_server->get_buffer(stream_type, cur_buf_idx);
    VisionIpcBufExtra extra = {
        cur_frame_data.frame_id,
        cur_frame_data.timestamp_sof,   // Start-of-Frame 时间戳（纳秒）
        cur_frame_data.timestamp_eof,   // End-of-Frame 时间戳（纳秒）
    };
    vipc_server->send(cur_yuv_buf, &extra);
}
```

### 4.2 关键参数

- **缓冲数量**: 18 个（`VIPC_BUFFER_COUNT`, `camera_common.h:10`）
- **传输方式**: 共享内存 + 信号量，零拷贝
- **每个缓冲携带**: frame_id, timestamp_sof, timestamp_eof

### 4.3 元数据消息

同时通过 cereal PubSub 发送 `roadCameraState` / `wideRoadCameraState` 消息，包含曝光信息、增益、灰度统计等元数据。

> 源码：`system/camerad/cameras/camera_qcom2.cc:224-255`

---

## 5. 阶段三：modeld 接收与帧同步

### 5.1 VisionIPC 客户端初始化

modeld 创建两个 VisionIPC 客户端分别接收两路相机流：

```python
# selfdrive/modeld/modeld.py:330-342
# 判断可用流：是否同时有 ROAD 和 WIDE_ROAD
use_extra_client = (VISION_STREAM_WIDE_ROAD in available_streams and
                    VISION_STREAM_ROAD in available_streams)

# 如果没有 ROAD 流，用 WIDE_ROAD 做主流
main_wide_camera = VISION_STREAM_ROAD not in available_streams

# 创建客户端
vipc_client_main = VisionIpcClient("camerad", main_stream, True, cl_context)
vipc_client_extra = VisionIpcClient("camerad", WIDE_ROAD, False, cl_context)
```

**在标准 TICI 硬件上**：
- `vipc_client_main` → Road Camera (`VISION_STREAM_ROAD`)
- `vipc_client_extra` → Wide Camera (`VISION_STREAM_WIDE_ROAD`)

### 5.2 帧时间同步

modeld 使用 `timestamp_sof`（帧起始时间戳）对齐两路相机帧：

```python
# selfdrive/modeld/modeld.py:388-414
# 主摄像头至少领先副摄像头 25ms
while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
    buf_main = vipc_client_main.recv()

# 副摄像头追赶到与主摄像头同步
while True:
    buf_extra = vipc_client_extra.recv()
    if meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
        break

# 时间差超过 10ms 则报错
if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
    cloudlog.error("frames out of sync!")
```

### 5.3 丢帧处理

如果检测到帧跳跃（`frame_id` 不连续），modeld 会执行 `prepare_only` 模式——仍然做透视变换以更新内部缓冲，但跳过模型前向推理：

```python
# selfdrive/modeld/modeld.py:447-457
vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
prepare_only = vipc_dropped_frames > 0
```

---

## 6. 阶段四：标定变换矩阵计算

这是整个流水线中最核心的数学部分。变换矩阵将原始相机像素坐标映射到模型期望的标准化视角。

### 6.1 坐标系定义

openpilot 使用三个主要坐标系：

| 坐标系 | X 轴 | Y 轴 | Z 轴 |
|--------|------|------|------|
| 设备坐标系 (device) | 前方 | 右方 | 下方 |
| 视图坐标系 (view) | 右方 | 下方 | 前方 |
| 标定坐标系 (calib) | 与设备相同，但经过标定旋转 |

设备坐标系与视图坐标系之间的变换：

```python
# common/transformations/camera.py:75-80
device_frame_from_view_frame = np.array([
    [ 0.,  0.,  1.],    # view_z(前方) → device_x(前方)
    [ 1.,  0.,  0.],    # view_x(右方) → device_y(右方)
    [ 0.,  1.,  0.]     # view_y(下方) → device_z(下方)
])
view_frame_from_device_frame = device_frame_from_view_frame.T
```

### 6.2 模型虚拟内参

模型期望接收的图像对应一个"虚拟相机"，其内参定义在 `common/transformations/model.py`：

**MED 模型**（用于 Road Camera 处理后的输入 `img`）：

```python
# common/transformations/model.py:10-18
MEDMODEL_INPUT_SIZE = (512, 256)    # 宽 x 高
MEDMODEL_CY = 47.6                  # Y 方向主点偏移（不在图像中心）
medmodel_fl = 910.0                  # 焦距（像素）

medmodel_intrinsics = np.array([
    [910.0,   0.0,  256.0],    # fx, 0, cx=512/2
    [  0.0, 910.0,   47.6],    # 0, fy, cy=47.6（偏上）
    [  0.0,   0.0,    1.0]
])
```

> **注意 MEDMODEL_CY = 47.6**：主点（光心）在图像非常靠上的位置（而非 128），这意味着模型关注的画面主要是地平线以下的区域（道路），上方天空区域被大幅裁剪。

**SBIG 模型**（用于 Wide Camera 处理后的输入 `big_img`）：

```python
# common/transformations/model.py:33-40
SBIGMODEL_INPUT_SIZE = (512, 256)
sbigmodel_fl = 455.0                 # 焦距为 MED 的一半 → 视角约 2 倍

sbigmodel_intrinsics = np.array([
    [455.0,   0.0,  256.0],    # fx, 0, cx=512/2
    [  0.0, 455.0,  151.8],    # 0, fy, cy=(256+47.6)/2=151.8
    [  0.0,   0.0,    1.0]
])
```

> SBIG 模型的 `cy = 151.8` 比 MED 的 `cy = 47.6` 大很多，表示广角图像的视野更均匀地分布在地平线上下。

### 6.3 变换矩阵计算过程

#### 6.3.1 预计算的逆标定矩阵

```python
# common/transformations/model.py:56-62
# 步骤 1: 构建"标定坐标系 → 模型像素坐标"的映射
medmodel_frame_from_calib_frame = medmodel_intrinsics @ get_view_frame_from_calib_frame(0, 0, 0, 0)
# 结果为 3x4 矩阵（3x3 旋转 + 3x1 平移）

# 步骤 2: 取 3x3 部分的逆（丢弃平移列，因为模型空间是纯投影）
calib_from_medmodel = np.linalg.inv(medmodel_frame_from_calib_frame[:, :3])
calib_from_sbigmodel = np.linalg.inv(sbigmodel_frame_from_calib_frame[:, :3])
```

#### 6.3.2 实时 warp 矩阵计算

```python
# common/transformations/model.py:65-70
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
    # 选择 MED 或 SBIG 的逆标定矩阵
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel

    # 欧拉角 [roll, pitch, yaw] → 3x3 旋转矩阵
    device_from_calib = rot_from_euler(device_from_calib_euler)

    # 完整变换链: 相机像素 ← 视图 ← 设备 ← 标定
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib

    # 最终: 相机像素 ← ... ← 模型坐标
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix  # 3x3 矩阵
```

**数学含义**：对于模型输出图像中的每个像素 `(u_model, v_model)`，变换矩阵告诉我们应该从原始相机图像的哪个位置 `(u_cam, v_cam)` 采样。这是一个**反向映射**（inverse mapping），在 OpenCL 内核中使用。

#### 6.3.3 在 modeld 主循环中的调用

```python
# selfdrive/modeld/modeld.py:429-436
if sm.updated["liveCalibration"]:
    device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
    dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]

    # Road Camera → MED 模型变换矩阵
    model_transform_main = get_warp_matrix(
        device_from_calib_euler,
        dc.fcam.intrinsics,    # Road Camera 内参 (f=2648)
        bigmodel_frame=False   # 使用 MED 模型内参 (f=910)
    ).astype(np.float32)

    # Wide Camera → SBIG 模型变换矩阵
    model_transform_extra = get_warp_matrix(
        device_from_calib_euler,
        dc.ecam.intrinsics,    # Wide Camera 内参 (f=567)
        bigmodel_frame=True    # 使用 SBIG 模型内参 (f=455)
    ).astype(np.float32)
```

变换的效果：
- **Road Camera** (f=2648) → MED 模型 (f=910)：缩小约 2.9 倍，使窄视角相机的图像适配模型的 512x256 输入。
- **Wide Camera** (f=567) → SBIG 模型 (f=455)：缩小约 1.25 倍，广角相机本身已经视角很大，微调即可。

---

## 7. 阶段五：OpenCL 透视变换

### 7.1 处理流程

从 Cython 层面调用 `DrivingModelFrame.prepare()`：

```python
# selfdrive/modeld/models/commonmodel_pyx.pyx:56-61
def prepare(self, VisionBuf buf, float[:] projection):
    # buf: VisionIPC 共享内存中的 NV12 帧（GPU 端 OpenCL 内存对象）
    # projection: 展平后的 3x3 变换矩阵（9 个 float）
    data = self.frame.prepare(buf.buf_cl, buf.width, buf.height,
                              buf.stride, buf.uv_offset, cprojection)
```

C++ 层面的实现（`commonmodel.cc:21-35`）：

```
1. run_transform()     — OpenCL 透视变换，分离 Y/U/V
2. 环形缓冲移位        — 保存历史帧
3. loadyuv_queue()     — Y 棋盘拆分 + UV 拷贝
4. copy_queue() × 2    — 组装历史帧 + 当前帧到最终缓冲
```

### 7.2 Y 通道变换

对 Y 平面执行透视变换，使用原始 warp 矩阵：

```
输入: NV12 帧的 Y 平面 (1928 x 1208, uint8)
    像素步长: 1 字节
    行步长: stride 字节

矩阵: projection_y = warp_matrix (3x3)

输出: 变换后的 Y 平面 (512 x 256, uint8)

OpenCL 工作项: 512 x 256 = 131,072 个并行线程
```

> 源码：`selfdrive/modeld/transforms/transform.cc:59-75`

### 7.3 U/V 通道变换

NV12 格式中 U 和 V 是交错存储的：`[U0, V0, U1, V1, ...]`。openpilot 巧妙地利用 `src_px_stride = 2` 和不同的 `src_offset` 来分别提取 U 和 V：

```
U 通道:
    输入: NV12 UV 平面，offset = uv_offset，像素步长 = 2（跳过 V）
    输入尺寸: 964 x 604（逻辑上 in_uv_width x in_uv_height）
    矩阵: projection_uv = transform_scale_buffer(warp_matrix, 0.5)
    输出: 256 x 128

V 通道:
    输入: NV12 UV 平面，offset = uv_offset + 1（跳到 V 的起始位置）
    其他同 U
    输出: 256 x 128
```

> 源码：`selfdrive/modeld/transforms/transform.cc:78-96`

### 7.4 UV 缩放矩阵

由于 UV 平面的分辨率是 Y 平面的一半，需要对变换矩阵进行缩放调整：

```c
// common/mat.h:69-85
mat3 transform_scale_buffer(const mat3 &in, float s) {
    // 公式: in_pt = (transform(out_pt/s + 0.5) - 0.5) * s
    // 当 s = 0.5 时：将 UV 的半分辨率坐标映射到 Y 的全分辨率坐标

    T_out = [[1/s, 0, 0.5],    // 输出坐标 → 全分辨率 + 像素中心偏移
             [0, 1/s, 0.5],
             [0,   0,   1]]

    T_in  = [[s, 0, -0.5*s],   // 全分辨率 → 半分辨率 + 反偏移
             [0, s, -0.5*s],
             [0, 0,      1]]

    return T_in @ warp_matrix @ T_out
}
```

### 7.5 透视变换内核（双线性插值）

OpenCL 内核对每个输出像素执行以下操作：

```c
// selfdrive/modeld/transforms/transform.cl
__kernel void warpPerspective(src, ..., dst, ..., M) {
    int dx = get_global_id(0);  // 输出像素 x（模型坐标）
    int dy = get_global_id(1);  // 输出像素 y（模型坐标）

    // 1. 透视映射：模型坐标 → 相机坐标（齐次坐标）
    float X0 = M[0]*dx + M[1]*dy + M[2];
    float Y0 = M[3]*dx + M[4]*dy + M[5];
    float W  = M[6]*dx + M[7]*dy + M[8];

    // 2. 齐次坐标归一化（含 1/32 精度的定点数化）
    W = (W != 0) ? 32.0 / W : 0;
    int X = round(X0 * W);
    int Y = round(Y0 * W);

    // 3. 分离整数部分（源像素索引）和小数部分（插值权重）
    int sx = X >> 5;          // 整数部分
    int sy = Y >> 5;
    float ax = (X & 31) / 32.0;  // 小数部分 [0, 1)
    float ay = (Y & 31) / 32.0;

    // 4. 读取 2x2 邻近像素（边界钳位）
    v0 = src[clamp(sy,   ...][clamp(sx,   ...)];
    v1 = src[clamp(sy,   ...][clamp(sx+1, ...)];
    v2 = src[clamp(sy+1, ...][clamp(sx,   ...)];
    v3 = src[clamp(sy+1, ...][clamp(sx+1, ...)];

    // 5. 双线性插值
    dst[dy][dx] = (1-ay)*(1-ax)*v0 + (1-ay)*ax*v1
                + ay*(1-ax)*v2     + ay*ax*v3;
}
```

这个内核对 Y、U、V 三个通道各执行一次（共 3 次 kernel dispatch）。

---

## 8. 阶段六：YUV 通道重排（loadyuv）

透视变换完成后，得到了 Y(512x256)、U(256x128)、V(256x128) 三个独立的平面。`loadyuv` 内核将它们重排为 6 通道格式。

### 8.1 Y 通道棋盘拆分

这是 openpilot 模型输入中比较独特的设计。512x256 的 Y 平面被按 2x2 块拆分为 4 个 128x256 的子通道：

```
原始 Y 平面 (512 宽 x 256 高):

    像素布局 (以 2x2 块为单位):
    ┌────┬────┐
    │ a  │ c  │  ← 偶数行
    ├────┼────┤
    │ b  │ d  │  ← 奇数行
    └────┴────┘
    偶数列  奇数列

拆分结果:
    通道 0 (y0): Y[偶数行, 偶数列]  → (128, 256)  ← 对应上面的 a
    通道 1 (y1): Y[奇数行, 偶数列]  → (128, 256)  ← 对应上面的 b
    通道 2 (y2): Y[偶数行, 奇数列]  → (128, 256)  ← 对应上面的 c
    通道 3 (y3): Y[奇数行, 奇数列]  → (128, 256)  ← 对应上面的 d
```

> **注意**：OpenCL 内核的 `loadys` 中注释标注的布局为 `02 / 13`（`loadyuv.cl:14-15`），意思是偶数行写入 y0 和 y2（左右分），奇数行写入 y1 和 y3（左右分）。

OpenCL 内核实现：

```c
// selfdrive/modeld/transforms/loadyuv.cl:3-29
__kernel void loadys(__global uchar8 const * const Y,
                     __global uchar * out, int out_offset) {
    const int gid = get_global_id(0);
    const uchar8 ys = Y[gid];  // 一次读 8 个像素

    // 偶数行: s0246 → y0(通道0), s1357 → y2(通道2)
    // 奇数行: s0246 → y1(通道1), s1357 → y3(通道3)
    if ((oy & 1) == 0) {
        outy0 = out + out_offset;                // y0
        outy1 = out + out_offset + UV_SIZE * 2;  // y2
    } else {
        outy0 = out + out_offset + UV_SIZE;      // y1
        outy1 = out + out_offset + UV_SIZE * 3;  // y3
    }
    vstore4(ys.s0246, 0, outy0 + ...);  // 偶数列像素
    vstore4(ys.s1357, 0, outy1 + ...);  // 奇数列像素
}
```

其中 `UV_SIZE = (512/2) * (256/2) = 256 * 128 = 32768` 字节，即每个子通道的大小。

### 8.2 U/V 通道直接拷贝

U 和 V 通道（各 256x128）直接拷贝到输出缓冲的通道 4 和通道 5：

```c
// selfdrive/modeld/transforms/loadyuv.cl:31-38
__kernel void loaduv(__global uchar8 const * const in,
                     __global uchar8 * out, int out_offset) {
    const int gid = get_global_id(0);
    out[gid + out_offset / 8] = in[gid];  // 直接拷贝
}
```

### 8.3 输出内存布局

单帧经过 loadyuv 后的内存布局：

```
偏移                    内容                    尺寸
0                       y0 (偶行偶列)           128 x 256 = 32768 字节
32768                   y1 (奇行偶列)           128 x 256 = 32768 字节
65536                   y2 (偶行奇列)           128 x 256 = 32768 字节
98304                   y3 (奇行奇列)           128 x 256 = 32768 字节
131072                  U                       128 x 256 = 32768 字节
163840                  V                       128 x 256 = 32768 字节
─────────────────────────────────────────────────
总计 = 6 x 32768 = 196608 字节 = MODEL_FRAME_SIZE (512 x 256 x 3/2)
```

可以理解为 `(6, 128, 256)` 的 uint8 张量。

---

## 9. 阶段七：时序帧组装与最终张量

### 9.1 环形缓冲与时序跳帧

模型接收两帧作为输入：**当前帧**和**历史帧**。两帧之间相隔固定的时间间隔。

```python
# selfdrive/modeld/constants.py:16-18
N_FRAMES = 2                # 时序输入帧数
MODEL_RUN_FREQ = 20         # 模型运行频率 (Hz)
MODEL_CONTEXT_FREQ = 5      # 策略网络上下文频率 (Hz)

# 计算:
# temporal_skip = MODEL_RUN_FREQ / MODEL_CONTEXT_FREQ - 1 = 20/5 - 1 = 3
# 时间间隔 = (temporal_skip + 1) / MODEL_RUN_FREQ = 4/20 = 0.2 秒
```

C++ 环形缓冲管理：

```cpp
// selfdrive/modeld/models/commonmodel.cc:21-34
cl_mem* DrivingModelFrame::prepare(...) {
    // 1. 对当前帧执行透视变换
    run_transform(yuv_cl, 512, 256, ...);

    // 2. 环形缓冲左移: 丢弃最老的帧，腾出空间给新帧
    for (int i = 0; i < temporal_skip; i++) {
        clEnqueueCopyBuffer(q, img_buffer_20hz_cl, img_buffer_20hz_cl,
            (i+1)*frame_size_bytes, i*frame_size_bytes, frame_size_bytes, ...);
    }
    // img_buffer_20hz_cl 容量 = (temporal_skip+1) * frame_size_bytes = 4 * 196608

    // 3. loadyuv: 将当前帧的 Y/U/V 重排到缓冲末尾 (last_img_cl)
    loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, last_img_cl);

    // 4. 组装最终输入: [历史帧(缓冲头部), 当前帧(缓冲尾部)]
    copy_queue(&loadyuv, q, img_buffer_20hz_cl, input_frames_cl,
               0, 0, frame_size_bytes);                     // 历史帧 → 前半
    copy_queue(&loadyuv, q, last_img_cl, input_frames_cl,
               0, frame_size_bytes, frame_size_bytes);       // 当前帧 → 后半

    return &input_frames_cl;
    // input_frames_cl 总大小 = buf_size = 2 * 196608 = 393216 字节
}
```

### 9.2 最终张量形状

```
input_frames_cl (393216 字节) 可解释为:

(1, 12, 128, 256) uint8

其中 12 通道 = 2帧 × 6通道/帧:

通道 0:  帧0 y0 (偶行偶列)
通道 1:  帧0 y1 (奇行偶列)
通道 2:  帧0 y2 (偶行奇列)
通道 3:  帧0 y3 (奇行奇列)
通道 4:  帧0 U
通道 5:  帧0 V
通道 6:  帧1 y0 (偶行偶列)    ← 当前帧
通道 7:  帧1 y1 (奇行偶列)
通道 8:  帧1 y2 (偶行奇列)
通道 9:  帧1 y3 (奇行奇列)
通道 10: 帧1 U
通道 11: 帧1 V

帧0 = 历史帧（0.2 秒前）
帧1 = 当前帧
```

### 9.3 送入模型

```python
# selfdrive/modeld/modeld.py:270-282
# 每个输入名调用 prepare() 得到 OpenCL 内存句柄
imgs_cl = {name: self.frames[name].prepare(bufs[name], transforms[name].flatten())
           for name in self.vision_input_names}

# TICI 上: 直接映射 OpenCL 显存为 tinygrad 张量（零拷贝）
if TICI and not USBGPU:
    for key in imgs_cl:
        if key not in self.vision_inputs:
            self.vision_inputs[key] = qcom_tensor_from_opencl_address(
                imgs_cl[key].mem_address,
                self.vision_input_shapes[key],  # (1, 12, 128, 256)
                dtype=dtypes.uint8)
else:
    # 非 TICI: 从 OpenCL 读回 CPU，构造 uint8 张量
    for key in imgs_cl:
        frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(...)
        self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8).realize()
```

**无显式归一化**：数据以原始 uint8 [0-255] 直接送入模型。归一化（如减均值、除标准差）由模型的第一层卷积隐式学习。

---

## 10. Road Camera 与 Wide Camera 对比

### 10.1 完整处理路径对比

| 处理阶段 | Road Camera | Wide Camera |
|---------|-------------|-------------|
| **传感器** | OX03C10 (1928x1208) | OX03C10 (1928x1208) |
| **物理焦距** | 8.0 mm（窄视角 ~24°） | 1.71 mm（广视角 ~120°） |
| **ISP 暗角校正** | 启用 | 不启用 |
| **软件焦距 (像素)** | 2648.0 | 567.0 |
| **VisionIPC 流** | `VISION_STREAM_ROAD` | `VISION_STREAM_WIDE_ROAD` |
| **VisionIPC 客户端** | `vipc_client_main` | `vipc_client_extra` |
| **目标模型** | MED 模型 | SBIG 模型 |
| **模型焦距** | 910.0 | 455.0 |
| **模型主点 cy** | 47.6 | 151.8 |
| **缩放比** | 2648/910 ≈ 2.9x | 567/455 ≈ 1.25x |
| **warp 矩阵函数** | `get_warp_matrix(..., bigmodel_frame=False)` | `get_warp_matrix(..., bigmodel_frame=True)` |
| **模型输入名** | `img` | `big_img` |
| **最终张量形状** | (1, 12, 128, 256) uint8 | (1, 12, 128, 256) uint8 |

### 10.2 输入选择逻辑

```python
# selfdrive/modeld/modeld.py:459-461
# 通过输入名中是否包含 "big" 来区分:
bufs = {
    name: buf_extra if 'big' in name else buf_main
    for name in model.vision_input_names
}
transforms = {
    name: model_transform_extra if 'big' in name else model_transform_main
    for name in model.vision_input_names
}
# model.vision_input_names 通常为 ['img', 'big_img']
```

### 10.3 视觉效果差异

由于 Road Camera 和 Wide Camera 的物理焦距和模型虚拟焦距不同，经过 warp 变换后：

- **Road Camera → img**: 来自窄视角相机的图像被"缩小"到 512x256，保留了远距离细节（如远处车辆、交通标志），但水平视角较小。模型内参 cy=47.6（很偏上）表示画面以地平线以下的道路为主。

- **Wide Camera → big_img**: 来自广角相机的图像被轻微缩小到 512x256，保留了宽广的侧向视野，有助于近距离和侧向感知（如相邻车道、交叉路口）。模型内参 cy=151.8 表示地平线大致居中。

---

## 11. 关键参数汇总

### 11.1 尺寸与格式

| 参数 | 值 | 来源 |
|------|-----|------|
| 原始传感器分辨率 | 1928 x 1208 | `ox03c10.cc:32-33` |
| ISP 输出格式 | NV12 (YUV420) | `hw.h:41, 53` |
| 模型输入尺寸 (Y) | 512 x 256 | `commonmodel.h:71-72` |
| 模型输入尺寸 (UV) | 256 x 128 | 半分辨率 |
| 单帧大小 | 196608 字节 | `512*256*3/2` |
| 最终张量形状 | (1, 12, 128, 256) | 2帧 × 6通道 |
| 数据类型 | uint8 [0-255] | 无归一化 |
| VisionIPC 缓冲数 | 18 | `camera_common.h:10` |

### 11.2 时序参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 模型运行频率 | 20 Hz | `constants.py:17` |
| 策略网络频率 | 5 Hz | `constants.py:18` |
| 时序帧数 | 2 | `constants.py:16` |
| 帧间隔 (temporal_skip) | 3 (即间隔 4 帧 = 0.2s) | 计算值 |

### 11.3 模型虚拟相机参数

| 参数 | MED 模型 (Road) | SBIG 模型 (Wide) |
|------|-----------------|------------------|
| 输入尺寸 | 512 x 256 | 512 x 256 |
| 焦距 fl | 910.0 | 455.0 |
| 主点 cx | 256.0 | 256.0 |
| 主点 cy | 47.6 | 151.8 |
| 对应输入名 | `img` | `big_img` |

### 11.4 物理相机参数（TICI AR/OX 配置）

| 参数 | Road Camera (fcam) | Wide Camera (ecam) |
|------|-------------------|-------------------|
| 分辨率 | 1928 x 1208 | 1928 x 1208 |
| 软件焦距 | 2648.0 像素 | 567.0 像素 |
| 物理焦距 | 8.0 mm | 1.71 mm |
| 暗角校正 | 启用 | 不启用 |

---

## 12. 附录：坐标系定义

### 12.1 三大坐标系的关系

```
               roll, pitch, yaw (rpyCalib)
标定坐标系 ──────────────────────────────> 设备坐标系
(calib)         rot_from_euler()            (device)
x:前, y:右, z:下                          x:前, y:右, z:下

                    转置矩阵
设备坐标系 ──────────────────────────────> 视图坐标系
(device)     view_frame_from_device_frame   (view)
x:前, y:右, z:下                          x:右, y:下, z:前

                   相机内参 K
视图坐标系 ──────────────────────────────> 相机像素坐标
(view)            intrinsics               (pixel)
x:右, y:下, z:前                          u:右, v:下
```

### 12.2 完整变换链（数学表达）

从模型像素坐标 `p_model` 到相机像素坐标 `p_camera`：

```
p_camera = K_camera @ T_view←device @ R_device←calib @ K_model_inv @ p_model

其中:
  K_camera     = 物理相机内参矩阵 (3x3)
  T_view←device = view_frame_from_device_frame (3x3 坐标系旋转)
  R_device←calib = rot_from_euler(rpyCalib) (3x3 标定旋转)
  K_model_inv  = calib_from_medmodel 或 calib_from_sbigmodel (3x3)

合并为:
  warp_matrix = K_camera @ T_view←device @ R_device←calib @ K_model_inv
```

在 OpenCL 内核中，对于输出图像中的每个像素 `(dx, dy)`：
1. 构造齐次坐标 `p_model = (dx, dy, 1)`
2. 计算 `p_camera = warp_matrix @ p_model`
3. 齐次归一化得到源图像坐标
4. 在源图像中做双线性插值采样

这就是 `cv2.WARP_INVERSE_MAP` 的含义——从输出坐标反向查找输入坐标。

---

## 13. FAQ

### Q1: "2.3 软件中的相机内参"中的相机内参是假设物理相机是理想的针孔相机，不考虑相机畸变？

**是的，openpilot 在整个图像处理流水线中使用的是纯针孔相机模型，不考虑镜头畸变。**

从 `CameraConfig` 的定义可以看出，每个相机仅存储三个参数：`width`、`height`、`focal_length`（`camera.py:8-25`）。由此生成的内参矩阵 K 是标准的 3x3 针孔模型矩阵，没有径向畸变（radial distortion）、切向畸变（tangential distortion）等参数。

在整个 `common/transformations/` 目录中，没有任何 `distort`/`undistort`/畸变校正相关的代码。warp 变换使用的是 3x3 单应性矩阵（homography），这只能表达投影变换，无法表达非线性的镜头畸变校正。

openpilot 的开发者也意识到这是一个近似。Wide Camera 的内参定义旁有注释：

```python
_ar_ox_fisheye = CameraConfig(1928, 1208, 567.0)
# focal length probably wrong? magnification is not consistent across frame
```

这说明 Wide Camera（广角/鱼眼镜头）的等效焦距在画面中心和边缘并不一致（这正是镜头畸变的表现），但 openpilot 仍然用单一焦距值来近似。

**这种近似之所以可行，有几个原因**：
1. **Road Camera**（f=8mm）是长焦镜头，畸变本身很小，针孔模型近似误差可忽略。
2. **Wide Camera**（f=1.71mm）虽然畸变较大，但模型输入的 SBIG 虚拟相机焦距（455）与物理焦距（567）接近，warp 变换的缩放幅度不大（~1.25x），且模型主要关注画面中心区域，边缘畸变的影响被弱化。
3. **模型通过训练学习了这种近似带来的残余畸变**——模型接收的就是"有畸变残余的虚拟针孔相机图像"，训练数据也同样如此，所以模型已经适应了这种输入特性。

### Q2: camerad 运行在 ISP（高通 IFE）中？不是 CPU/GPU？

**ISP 处理和 camerad 进程是分开的，各自负责不同的工作。**

具体来说：

- **ISP（IFE/BPS）是高通 SoC 中的专用硬件加速器**，独立于 CPU 和 GPU。图像的核心处理（黑电平校正、去拜耳、白平衡、色彩校正、伽马、YUV 转换等——即第 3 章所述的全部 ISP 流水线）都在这个专用硬件上执行，不消耗 CPU 或 GPU 资源。

- **camerad 是运行在 CPU 上的用户态进程**（`system/camerad/`），它的职责是：
  1. **配置** ISP 硬件：通过 V4L2/ioctl 接口设置 ISP 寄存器（IFE 模块参数、色彩矩阵、伽马 LUT 等），见 `spectra.cc:702-898`。
  2. **配置**传感器：通过 I2C 写入传感器寄存器（曝光、增益、PLL 等），见 `ox03c10.cc:110-127`。
  3. **事件循环**：通过 `poll()` + `VIDIOC_DQEVENT` 等待 ISP 硬件完成每帧处理（`camera_qcom2.cc:257-323`）。
  4. **自动曝光（AE）**：在 CPU 上读取 Y 平面亮度、计算目标曝光参数（`camera_qcom2.cc:126-222`）。
  5. **帧发送**：将 ISP 输出的 YUV 帧通过 VisionIPC 共享内存发送给下游进程（`camera_common.cc:45-61`）。

所以数据流是：

```
传感器 → ISP 硬件 (IFE) → YUV NV12 (硬件输出到共享内存)
                ↑ 配置/控制               ↓ 通知帧完成
           camerad (CPU)              camerad (CPU)
                                         ↓ VisionIPC 发送
                                      modeld 等下游进程
```

camerad 本身不做任何像素级的图像处理，它只是 ISP 硬件的"管理者"。

### Q3: 相机输出的帧率由谁控制？Sensor？camerad？

**帧率由传感器硬件的寄存器配置决定，camerad 在初始化时通过 I2C 写入这些寄存器。**

OX03C10 传感器的帧率由以下寄存器决定（`ox03c10_registers.h`）：

```c
// PLL 设置决定像素时钟频率
{0x0303, 0x01},                      // pll1_prediv
{0x0304, 0x01}, {0x0305, 0x2c},      // pll1_loopdiv = 300

// 行时序（水平总像素数）
{0x380c, 0x04}, {0x380d, 0x47},      // HTS = 0x447 = 1095

// 帧时序（垂直总行数）
{0x380e, 0x08}, {0x380f, 0x15},      // VTS = 0x815 = 2069
```

帧率 = 像素时钟 / (HTS × VTS)。寄存器注释提到基础配置为 `60fps_HDR4_LFR`（60fps HDR 低帧率模式），但实际 VTS 被调大了（从注释中的 `0x2ae` 改为 `0x815`），对应约 53.65ms 的帧周期。

此外，传感器配置了 **FSIN（Frame Sync，帧同步）外部触发**模式（`ox03c10_registers.h:67-69`）：

```c
// FSIN (frame sync) with external pulses
{0x3009, 0x2},
{0x3015, 0x2},
```

这意味着传感器并非自由运行（free-running），而是等待外部同步脉冲来触发每帧的开始。comma 3X 硬件提供 **20 Hz** 的 FSIN 脉冲信号，所以三个相机被精确同步到 20 fps。

从 cereal 服务注册表也能确认：

```python
# cereal/services.py
"roadCameraState": (True, 20., 20),
"wideRoadCameraState": (True, 20., 20),
```

**总结**：camerad 在启动时通过 I2C 配置传感器的 PLL、时序和 FSIN 模式，之后帧率由硬件 FSIN 脉冲驱动（20 Hz），camerad 只是被动等待帧事件。

### Q4: 相机图像的 warp 处理是否改变了物体在图像中的长宽比？warp 前后，相机图像 FOV 是否改变？

**长宽比**：在画面中心附近基本不变，边缘可能有轻微变化。

warp 变换的核心矩阵是：

```
warp_matrix = K_camera @ T_view←device @ R_device←calib @ K_model_inv
```

由于物理相机和模型虚拟相机的内参矩阵 K 都是**各向同性的**（fx = fy），所以变换在 x 和 y 方向上的缩放比例相同。对于画面中心附近的物体，这等同于均匀缩放，**长宽比完全保持不变**。

对于远离光轴的物体，由于标定旋转 R 的存在（以及透视投影本身的非线性），会引入轻微的透视变形，但这不是"长宽比改变"——而是正常的透视效果（近大远小）。

**FOV（视场角）**：显著改变。warp 后的 FOV 完全由模型虚拟相机的内参决定，而非物理相机。

计算水平半视场角 θ = arctan(cx / f)，可得：

| 相机/模型 | 焦距 (px) | cx (px) | 水平 FOV |
|-----------|-----------|---------|----------|
| Road Camera 物理 | 2648 | 964 | 2 × arctan(964/2648) ≈ **40°** |
| MED 模型虚拟 | 910 | 256 | 2 × arctan(256/910) ≈ **31°** |
| Wide Camera 物理 | 567 | 964 | 2 × arctan(964/567) ≈ **119°** |
| SBIG 模型虚拟 | 455 | 256 | 2 × arctan(256/455) ≈ **59°** |

可以看出：
- **Road Camera**: 物理 FOV ~40° → 模型 FOV ~31°，变窄了约 23%。模型只取了中心区域的精华部分。
- **Wide Camera**: 物理 FOV ~119° → 模型 FOV ~59°，大幅缩窄。广角相机的边缘大畸变区域被裁掉，只保留中心约一半的视野。

注意垂直方向上 FOV 的变化更大，因为模型的 cy 偏移很不对称（尤其是 MED 模型的 cy=47.6 远离图像中心 128），这意味着模型在垂直方向上主要关注地平线以下的道路区域。

### Q5: 输入给神经网络模型的图像是虚拟相机图像，这个图像是否统一了相机 RPY 姿态？

**是的，这正是 warp 变换的核心目的之一。**

warp 矩阵中包含标定旋转：

```python
device_from_calib = rot_from_euler(device_from_calib_euler)  # rpyCalib → 3x3 旋转
camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
warp_matrix = camera_from_calib @ calib_from_model
```

`rpyCalib`（roll, pitch, yaw）描述的是设备坐标系相对于标定坐标系的旋转偏差。标定坐标系可以理解为一个**规范化的参考姿态**。当 warp 矩阵中纳入 `device_from_calib` 这个旋转后，变换的效果是：

**将实际相机图像"纠正"到标定坐标系对应的标准视角下。**

具体来说：
- 不同车辆上的 comma 3X 安装角度不完全一样（可能有几度的 pitch/yaw 偏差）。
- `calibrationd` 进程通过观测消失点等方法估算出这个安装偏差 `rpyCalib`。
- warp 变换将这个偏差旋转掉，使得**无论相机实际安装在什么角度，模型看到的图像都像是从同一个标准姿态拍摄的**。

这对模型泛化至关重要：
1. **训练时**：所有训练数据都经过了相同的标定纠正，模型学到的是标准视角下的特征。
2. **推理时**：每辆车的安装偏差被实时补偿，模型始终工作在熟悉的"标准视角"下。
3. **效果**：模型不需要学习处理各种安装角度的变化，可以专注于理解道路场景。

如果不做标定纠正（rpyCalib = [0,0,0]），相机 pitch 偏差几度就会导致地平线在图像中偏移几十个像素，这对车道线检测等任务会造成显著影响。


*本文档基于 openpilot 源码分析生成，不包含 `tools/dashcam` 目录的代码。*
