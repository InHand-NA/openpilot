# dashcam 独立感知模块实现方案

## 1. 需求分析

根据 task.md，开发 `tools/dashcam` 模块，实现以下功能：

| 需求 | 说明 |
|------|------|
| 独立运行 | 不依赖 openpilot 的进程管理器、cereal 消息总线；可导入 openpilot 工具模块 |
| Carla 世界 | 初始化 Carla 仿真环境，启用 Carla 自动驾驶 |
| 传感器采集 | 从 Carla 采集双路相机图像和车辆状态 |
| 姿态标定 | 支持已知姿态模式和在线标定模式（参考 calibrationd） |
| 透视变换 | OpenCL 加速的图像透视变换（复用 openpilot 的 DrivingModelFrame） |
| 视觉推理 | GPU 推理 openpilot 视觉网络（参考 modeld 实现） |
| 不需要策略网络 | 仅用视觉网络，不做行驶规划 |
| 可视化 | 在原始图像上叠加车道线、路沿线、前车检测等感知结果 |
| 视频输出 | 支持将可视化结果保存为视频文件 |

## 2. 架构设计

### 2.1 总体架构

```
┌────────────────────────────────────────────────────────────────────┐
│                      tools/dashcam/run.py                          │
│                         (主循环)                                    │
│                                                                    │
│  ┌──────────────┐  ┌───────────────────┐  ┌─────────────────────┐  │
│  │  CarlaWorld   │  │   VisionModel     │  │    Visualizer       │  │
│  │              │→→│                   │→→│                     │  │
│  │ - 双路相机    │  │ - RGB→NV12 (CL)  │  │ - 车道线 (4条)      │  │
│  │ - autopilot  │  │ - warpPerspective │  │ - 路沿线 (2条)      │  │
│  │ - NPC 车辆   │  │ - loadYUV (CL)   │  │ - 前车检测          │  │
│  │ - 车辆状态    │  │ - tinygrad 推理   │  │ - 运动估计          │  │
│  └──────────────┘  │ - 输出解析        │  │ - 标定状态          │  │
│        ↑           └───────────────────┘  │ - 视频写入          │  │
│        │                    ↑              └─────────────────────┘  │
│        │            ┌───────┴────────┐                             │
│        │            │   Calibrator   │                             │
│        └────────────│ - 已知姿态模式  │                             │
│       (vehicle_speed)│ - 在线标定模式  │                             │
│                     └────────────────┘                             │
└────────────────────────────────────────────────────────────────────┘
            ↕ (Carla Python API)
    ┌───────────────────┐
    │   Carla Server     │
    └───────────────────┘
```

### 2.2 数据流

```
Carla 相机回调 (RGB 1928×1208, 20Hz)
  ↓
RGB → NV12 (OpenCL: rgb_to_nv12.cl, 复用 sim/camerad 的方式)
  → NV12 格式 cl_mem, 含 Y 平面 + 交织 UV 平面
  ↓
warpPerspective (OpenCL: transform.cl, 由 DrivingModelFrame.prepare 驱动)
  → 透视变换 + 色彩通道分离 (Y/U/V 各自变换)
  → 输出到模型尺寸 512×256
  ↓
loadYUV (OpenCL: loadyuv.cl, 由 DrivingModelFrame.prepare 驱动)
  → Y 拆分为 4 子采样通道 + U + V = 6 通道
  → 与历史帧拼接为 12 通道 (temporal_skip=3)
  → (1, 12, 128, 256) uint8 cl_mem
  ↓
Tinygrad 视觉网络 GPU 推理
  → 输入: img (1,12,128,256) + big_img (1,12,128,256)
  → 输出: 1576 维向量
  ↓
输出解析 (Parser: softmax/sigmoid/MDN 解码)
  → 车道线、路沿、前车、相机位姿、元事件
  ↓
在线标定更新 (从 pose 输出反推 pitch/yaw, 参考 calibrationd)
  ↓
可视化叠加 (OpenCV 绘制到原始图像, 缩放到 964×604 显示)
  → 窗口显示 + 可选视频文件写入
```

### 2.3 模块依赖关系

**复用的 openpilot 编译模块**:
- `selfdrive/modeld/models/commonmodel_pyx.so` — `CLContext`, `DrivingModelFrame`（OpenCL 预处理）
- `selfdrive/modeld/parse_model_outputs.Parser` — 输出解析
- `selfdrive/modeld/constants.ModelConstants` — 常量定义
- `common/transformations/model.get_warp_matrix` — 变换矩阵计算
- `common/transformations/camera` — 坐标系定义、相机参数
- `common/transformations/orientation` — 旋转矩阵/欧拉角转换

**不使用的 openpilot 运行时基础设施**:
- ❌ `cereal.messaging` (PubMaster/SubMaster) — 无进程间消息通信
- ❌ `msgq.visionipc` (VisionIpcServer/Client) — 无共享内存图像传输
- ❌ `openpilot.common.params` (Params) — 无参数数据库
- ❌ openpilot 进程管理器 (manager)
- ❌ 策略网络 (`driving_policy_tinygrad.pkl`)

**第三方依赖**:
- `carla` — Carla Python API
- `tinygrad` — 神经网络 GPU 推理
- `pyopencl` — RGB→NV12 转换
- `opencv-python` — 可视化与视频输出
- `numpy` — 数值计算

## 3. 模块设计

### 3.1 文件结构

```
tools/dashcam/
├── __init__.py
├── run.py                # 主入口与主循环
├── carla_world.py        # Carla 世界管理（独立于 openpilot 运行时）
├── vision_model.py       # 视觉模型加载、OpenCL 预处理、GPU 推理、解析
├── calibrator.py         # 已知姿态 + 在线标定（参考 calibrationd）
└── visualizer.py         # OpenCV 可视化 + 视频写入
```

### 3.2 各模块详细设计

#### `carla_world.py` — Carla 世界管理

参考现有 `tools/sim/bridge/carla/carla_world.py`，去除 openpilot 运行时依赖。

```python
class DashcamCarlaWorld:
    """独立的 Carla 世界管理"""

    def __init__(self, host, port, town, spawn_point, camera_pitch_deg, camera_yaw_deg,
                 high_quality=False, num_npc=20):
        # 1. 连接 Carla, 加载地图
        # 2. 配置同步模式: fixed_delta_seconds=0.01, actor_active_distance=150.0
        # 3. 生成主车 (Tesla), 配置物理参数
        # 4. 启用 Carla autopilot (含 TM retry 逻辑)
        # 5. 创建窄角相机 (FOV=40) + 广角相机 (FOV=120)
        #    - 分辨率 1928×1208, sensor_tick=1/20
        #    - transform: Location(x=0.8, z=1.13), Rotation(pitch, yaw)
        # 6. 创建 IMU 传感器 (sensor_tick=0.01)
        # 7. Traffic Manager + NPC 车辆 + dormant respawn

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """获取最新相机 RGB 帧 (主相机, 广角), 无新帧返回 None"""

    def get_vehicle_speed(self) -> float:
        """获取 3D 速度标量 (m/s)"""

    def tick(self):
        """推进仿真一步 (0.01s)"""

    def close(self):
        """销毁所有 actor, 关闭 TM"""
```

**关键设计点**:

| 参数 | 值 | 说明 |
|------|-----|------|
| fixed_delta_seconds | 0.01 | 与 openpilot sim bridge 一致 |
| sensor_tick (camera) | 1/20 = 0.05 | 每 5 个仿真步产生一帧 |
| sensor_tick (IMU) | 0.01 | 每步一条 IMU 数据 |
| TICKS_PER_FRAME | 5 | 主循环每 5 次 tick 检查一帧 |
| W, H | 1928, 1208 | 与 openpilot 标准一致 |

相机回调使用 threading.Lock 保护帧缓冲区，主循环通过 `_new_frame` 标志判断是否有新帧到达（与现有 sim bridge 的同步帧机制一致）。

#### `vision_model.py` — 视觉模型与 OpenCL 预处理

核心设计：**复用 openpilot 的 C++/OpenCL 预处理管线，通过扩展 Cython 接口绕开 VisionBuf 依赖**。

##### OpenCL 预处理接口

现有 `DrivingModelFrame.prepare()` 签名接受 `VisionBuf`（与 VisionIPC 绑定）。为独立使用，需扩展 `commonmodel_pyx.pyx` 添加一个接受原始 NV12 数据的方法：

```python
# 新增到 commonmodel_pyx.pyx 的 ModelFrame 类:
def prepare_from_yuv(self, CLContext context,
                     cnp.ndarray[cnp.uint8_t, ndim=1] yuv_data,
                     int width, int height, int stride, int uv_offset,
                     float[:] projection):
    """从原始 NV12 数据准备模型输入，无需 VisionBuf/VisionIPC

    参数:
      context: CLContext (提供 OpenCL context)
      yuv_data: NV12 格式的 YUV 数据 (1D uint8 array)
      width, height, stride, uv_offset: 图像几何参数
      projection: 3×3 透视变换矩阵 (展平为 9 元素 float32 数组)

    返回:
      CLMem 对象 (指向 OpenCL 内存中的模型输入)
    """
    cdef mat3 cprojection
    memcpy(cprojection.v, &projection[0], 9 * sizeof(float))
    cdef cl_int err
    # 在 CLContext 的 OpenCL context 中创建临时 cl_mem
    cdef cl_mem yuv_cl = clCreateBuffer(
        context.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        yuv_data.shape[0],
        <void*>&yuv_data[0], &err)
    cdef cl_mem * data
    data = self.frame.prepare(yuv_cl, width, height, stride, uv_offset, cprojection)
    clReleaseMemObject(yuv_cl)
    return CLMem.create(data)
```

此方法在 `CLContext` 的 OpenCL context 内创建临时 `cl_mem`，调用 C++ `DrivingModelFrame::prepare()` 执行完整的 OpenCL 预处理流水线（warpPerspective → loadYUV → 时序帧拼接），然后释放临时缓冲区。

##### RGB → NV12 转换

复用 `tools/sim/rgb_to_nv12.cl` 内核，通过 pyopencl 驱动：

```python
class RGBToNV12Converter:
    """使用 OpenCL 将 RGB 转换为 NV12 格式"""

    def __init__(self, width=1928, height=1208):
        # 加载并编译 rgb_to_nv12.cl 内核
        # 创建 pyopencl context 和 command queue

    def convert(self, rgb: np.ndarray) -> np.ndarray:
        """RGB (H,W,3) uint8 → NV12 bytes"""
```

**注意**: pyopencl 和 CLContext 使用**不同的 OpenCL context**。RGB→NV12 结果需要通过 CPU 内存中转（numpy 数组），然后由 `prepare_from_yuv()` 上传到 CLContext 的 OpenCL context。这是必要的权衡——两个 context 的 cl_mem 不可共享。

##### VisionModel 类

```python
class VisionModel:
    """视觉网络加载、预处理与推理"""

    def __init__(self):
        # 1. 选择推理后端
        os.environ.setdefault('DEV', 'GPU')  # GPU 优先，fallback 到 CPU
        # 2. 初始化 CLContext 和 DrivingModelFrame (×2: img + big_img)
        self.cl_context = CLContext()
        self.frames = {name: DrivingModelFrame(self.cl_context, temporal_skip)
                       for name in vision_input_names}
        # 3. 初始化 RGB→NV12 转换器
        self.rgb_converter = RGBToNV12Converter()
        # 4. 加载 tinygrad 视觉模型 + 元数据
        # 5. 初始化 Parser

    def preprocess(self, rgb_main, rgb_wide, intrinsics, rpyCalib):
        """图像预处理 (OpenCL 加速)

        流程:
        1. rgb_converter.convert(rgb) → NV12 numpy
        2. get_warp_matrix() → 变换矩阵
        3. frame.prepare_from_yuv(nv12, ..., warp_matrix) → cl_mem
        4. frame.buffer_from_cl(cl_mem) → numpy → Tensor
        返回: {img: Tensor, big_img: Tensor}
        """

    def run(self, inputs) -> dict:
        """vision_run(**inputs) → 解析后的输出字典"""

    def get_camera_odometry(self, parsed) -> dict:
        """提取 pose 相关输出"""
```

##### NV12 格式说明

`rgb_to_nv12.cl` 输出的 NV12 内存布局：
```
偏移 0:           Y 平面 (W × H 字节, stride = W)
偏移 W*H:         UV 交织平面 (W × H/2 字节)
                  [U0 V0 U1 V1 ...]
```

对应 `DrivingModelFrame.prepare()` 的参数：
- `width = 1928`, `height = 1208`
- `stride = 1928`（Y 行步长）
- `uv_offset = 1928 * 1208`（UV 平面起始偏移）

#### `calibrator.py` — 标定

提供两种标定模式，通过命令行参数选择。先实现已知姿态模式，再实现在线标定。

##### 模式 1: 已知姿态 (KnownPoseCalibrator)

用于调试和验证，直接从 Carla 相机安装角度设置 rpyCalib。

```python
class KnownPoseCalibrator:
    """已知相机姿态的标定器"""

    def __init__(self, pitch_deg: float, yaw_deg: float, height: float = 1.22):
        self.rpy = np.array([0.0, np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)])
        self.height = height
        self.cal_status = 'calibrated'
        self.valid_blocks = 50  # 直接标记为已标定

    @property
    def rpyCalib(self) -> np.ndarray:
        return self.rpy

    @property
    def calibrated(self) -> bool:
        return True

    def update(self, vision_output: dict, vehicle_speed: float):
        pass  # 无需更新
```

当 `--perfect-cam` 时，pitch=0, yaw=0；否则使用 `--camera-pitch` 和 `--camera-yaw` 的值。

##### 模式 2: 在线标定 (OnlineCalibrator)

**直接移植 calibrationd.py 的 `Calibrator` 核心算法**，去除 cereal/Params 依赖：

```python
class OnlineCalibrator:
    """在线姿态标定，参考 selfdrive/locationd/calibrationd.py

    核心算法:
    1. 条件门控: 车速 > MIN_SPEED_FILTER 且偏航角速度 < MAX_YAW_RATE_FILTER
    2. 从 cameraOdometry.trans 反推 observed_rpy:
       - pitch = -arctan2(trans[2], trans[0])
       - yaw   =  arctan2(trans[1], trans[0])
    3. 与历史 rpy 融合: new_rpy = rot(smooth_rpy) × rot(observed_rpy)
    4. 以线性衰减权重写入滑动 block 缓存
    5. 多 block 取均值，判定标定状态
    """

    # 关键常量 (与 calibrationd 一致)
    MIN_SPEED_FILTER = 15 * 0.44704   # 15 MPH → m/s
    MAX_YAW_RATE_FILTER = np.radians(2)
    MAX_VEL_ANGLE_STD = np.radians(0.25)
    BLOCK_SIZE = 100
    INPUTS_NEEDED = 5
    INPUTS_WANTED = 50
    SMOOTH_CYCLES = 10
    PITCH_LIMITS = np.array([-0.09074112, 0.17])
    YAW_LIMITS = np.array([-0.06912048, 0.06912048])

    def __init__(self, pitch_deg_init=0.0, yaw_deg_init=0.0, height_init=1.22):
        self.rpy = np.array([0.0, np.deg2rad(pitch_deg_init), np.deg2rad(yaw_deg_init)])
        self.height = np.array([height_init])
        # 滑动窗口缓存 (与 calibrationd 完全一致)
        self.rpys = np.tile(self.rpy, (self.INPUTS_WANTED, 1))
        self.heights = np.tile(self.height, (self.INPUTS_WANTED, 1))
        self.wide_from_device_eulers = np.zeros((self.INPUTS_WANTED, 3))
        self.valid_blocks = 0
        self.block_idx = 0
        self.idx = 0
        self.old_rpy = np.zeros(3)
        self.old_rpy_weight = 0.0
        self.v_ego = 0.0
        self.cal_status = 'uncalibrated'

    @property
    def rpyCalib(self) -> np.ndarray:
        return self.get_smooth_rpy()

    @property
    def calibrated(self) -> bool:
        return self.cal_status == 'calibrated'

    def update(self, vision_output: dict, vehicle_speed: float):
        """接收视觉网络输出，更新标定

        vision_output 需包含:
        - pose: (1, 6) [trans_x, trans_y, trans_z, rot_r, rot_p, rot_y]
        - pose_stds: (1, 6)
        - wide_from_device_euler: (1, 3)
        - road_transform: (1, 6)
        - road_transform_stds: (1, 6)
        """
        self.v_ego = vehicle_speed
        # 提取 cameraOdometry 等效数据
        trans = vision_output['pose'][0, :3].tolist()
        rot = vision_output['pose'][0, 3:].tolist()
        trans_std = vision_output['pose_stds'][0, :3].tolist()
        wide_euler = vision_output['wide_from_device_euler'][0].tolist()
        road_trans = vision_output['road_transform'][0, :3].tolist()
        road_trans_std = vision_output['road_transform_stds'][0, :3].tolist()
        # 调用 handle_cam_odom() — 移植自 calibrationd.Calibrator
        self.handle_cam_odom(trans, rot, wide_euler, trans_std, road_trans, road_trans_std)

    def handle_cam_odom(self, trans, rot, wide_from_device_euler,
                        trans_std, road_transform_trans, road_transform_trans_std):
        """与 calibrationd.py Calibrator.handle_cam_odom 逻辑一致"""
        # ... (移植核心算法)

    def get_smooth_rpy(self):
        """平滑过渡 (与 calibrationd 一致)"""
        if self.old_rpy_weight > 0:
            return self.old_rpy_weight * self.old_rpy + (1.0 - self.old_rpy_weight) * self.rpy
        return self.rpy

    def update_status(self):
        """更新标定状态 (与 calibrationd 一致)"""
        # uncalibrated / calibrated / invalid / recalibrating
```

#### `visualizer.py` — 可视化与视频输出

```python
class Visualizer:
    """在 Carla 图像上叠加感知结果，支持窗口显示和视频写入"""

    DISPLAY_SCALE = 0.5  # 窗口缩放到 964×604

    def __init__(self, save_video_path: str = '', no_display: bool = False,
                 source_fps: float = 20.0):
        # 初始化 cv2.VideoWriter (若 save_video_path 非空)
        # 初始化显示窗口

    def draw(self, frame_rgb: np.ndarray, vision_output: dict,
             intrinsics: np.ndarray, rpyCalib: np.ndarray,
             camera_height: float, vehicle_speed: float,
             calibrator_status: str, fps: float) -> np.ndarray:
        """在帧上绘制所有感知结果"""
        # 转换 RGB→BGR (OpenCV 格式)
        # 1. 绘制车道线 (4条)
        # 2. 绘制路沿线 (2条)
        # 3. 绘制前车检测
        # 4. 绘制信息面板
        # 缩放到 DISPLAY_SCALE
        # 写入视频 (若启用)
        # 显示窗口 (若启用)

    def close(self):
        """释放 VideoWriter 和窗口"""
```

**绘制细节**:

| 元素 | 数据来源 | 颜色 | 说明 |
|------|---------|------|------|
| 车道线 (左外/内) | lane_lines[0,1], lane_lines_prob[0,1] | 绿色 | 透明度=置信度 |
| 车道线 (右内/外) | lane_lines[2,3], lane_lines_prob[2,3] | 橙黄色 | 透明度=置信度 |
| 路沿线 (左/右) | road_edges[0,1] | 红色 | 标记道路边界 |
| 前车 | lead[0], lead_prob[0] | 青色圆圈 | 标注距离和相对速度 |
| 信息面板 | 多源 | 白色文字 | 左上角半透明背景 |

**3D→2D 投影**:

```python
def project_to_image(xs, ys, zs, intrinsics, rpyCalib, height):
    """道路坐标 → 图像像素"""
    from openpilot.common.transformations.camera import get_view_frame_from_road_frame
    v_from_road = get_view_frame_from_road_frame(0.0, rpyCalib[1], rpyCalib[2], height)
    pts = np.stack([xs, ys, zs, np.ones_like(xs)], axis=0)  # 4×N
    C = intrinsics @ v_from_road  # 3×4 投影矩阵
    uvw = C @ pts
    uv = uvw[:2] / np.clip(uvw[2:3], 1e-6, None)
    return uv.T  # N×2
```

**信息面板内容**:
```
FPS: 18.3
Speed: 12.3 m/s (44.3 km/h)
Calib: pitch=4.98° yaw=2.95° [CALIBRATED]
Calib blocks: 12/5
Height: 1.22m
Lead: 32.5m  v_rel=-2.1 m/s  prob=0.95
Cam Odom: vx=12.1 vy=0.02 vz=-0.15 m/s
```

**窗口缩放**: 原始帧 1928×1208 → 缩放 0.5 → 显示窗口 964×604。视频文件写入原始分辨率。

#### `run.py` — 主入口

```python
TICKS_PER_FRAME = 5  # 与 openpilot sim bridge 一致

def main():
    parser = argparse.ArgumentParser(description='Standalone dashcam with openpilot vision')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', default='Town04_Opt')
    parser.add_argument('--spawn-point', type=int, default=16)
    parser.add_argument('--camera-pitch', type=float, default=5.0)
    parser.add_argument('--camera-yaw', type=float, default=3.0)
    parser.add_argument('--camera-height', type=float, default=1.13)
    parser.add_argument('--perfect-cam', action='store_true',
                        help='已知姿态模式: pitch=0, yaw=0, 跳过标定')
    parser.add_argument('--num-npc', type=int, default=20)
    parser.add_argument('--high-quality', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--save-video', type=str, default='',
                        help='保存可视化结果到视频文件 (mp4)')
    args = parser.parse_args()

    # 初始化
    pitch = 0.0 if args.perfect_cam else args.camera_pitch
    yaw = 0.0 if args.perfect_cam else args.camera_yaw

    world = DashcamCarlaWorld(
        host=args.host, port=args.port, town=args.town,
        spawn_point=args.spawn_point,
        camera_pitch_deg=pitch, camera_yaw_deg=yaw,
        high_quality=args.high_quality, num_npc=args.num_npc)

    model = VisionModel()

    if args.perfect_cam:
        calibrator = KnownPoseCalibrator(pitch_deg=0.0, yaw_deg=0.0,
                                         height=args.camera_height)
    else:
        calibrator = KnownPoseCalibrator(pitch_deg=pitch, yaw_deg=yaw,
                                         height=args.camera_height)
        # 第二阶段切换为 OnlineCalibrator

    intrinsics = ...  # DEVICE_CAMERAS[("pc", "unknown")].fcam.intrinsics
    visualizer = Visualizer(save_video_path=args.save_video,
                            no_display=args.no_display)

    # 主循环
    tick_count = 0
    while True:
        world.tick()
        tick_count += 1

        if tick_count % TICKS_PER_FRAME != 0:
            continue

        road_rgb, wide_rgb = world.get_frame()
        if road_rgb is None:
            continue

        # 预处理 + 推理
        inputs = model.preprocess(road_rgb, wide_rgb, intrinsics, calibrator.rpyCalib)
        output = model.run(inputs)

        # 更新标定
        calibrator.update(output, world.get_vehicle_speed())

        # 可视化
        visualizer.draw(road_rgb, output, intrinsics, calibrator.rpyCalib,
                        calibrator.height, world.get_vehicle_speed(),
                        calibrator.cal_status, fps)

    world.close()
    visualizer.close()
```

## 4. 关键技术决策

### 4.1 仿真频率（与 openpilot sim bridge 一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| fixed_delta_seconds | 0.01 | 仿真步长 100Hz |
| camera sensor_tick | 1/20 = 0.05 | 每 5 步产生一帧 |
| TICKS_PER_FRAME | 5 | 主循环每 5 次 tick 处理一帧 |
| 有效帧率 | 20 Hz | 与 modeld 的 MODEL_RUN_FREQ 一致 |
| IMU sensor_tick | 0.01 | 每步一条（预留给在线标定） |

### 4.2 OpenCL 预处理管线

完整复用 openpilot 生产代码中的 OpenCL 管线：

```
                    commonmodel_pyx.so
                   ┌──────────────────────────────┐
NV12 numpy ──────→│ prepare_from_yuv()            │
                   │  ├─ clCreateBuffer (临时)      │
                   │  ├─ transform_queue() ──────── │──→ warpPerspective (transform.cl)
                   │  │    Y/U/V 各一次 dispatch    │
                   │  ├─ CopyBuffer (时序帧移位)    │
                   │  ├─ loadyuv_queue() ────────── │──→ loadys + loaduv (loadyuv.cl)
                   │  ├─ copy_queue (双帧拼接)      │
                   │  └─ clFinish                   │
                   │                                │
                   │  返回 cl_mem (12×128×256 uint8) │
                   └──────────────────────────────┘
                            │
                            ▼
                   buffer_from_cl() → numpy → Tensor
```

需要对 `commonmodel_pyx.pyx` 添加 `prepare_from_yuv()` 方法（约 20 行 Cython 代码），并重新编译：
```bash
scons selfdrive/modeld/models/commonmodel_pyx.so
```

### 4.3 GPU 推理

参考 `modeld.py` 的推理方式：

```python
# 设备选择逻辑 (参考 modeld.py)
from openpilot.system.hardware import TICI
os.environ['DEV'] = 'QCOM' if TICI else 'CPU'
# 若检测到可用 GPU，可设置 'CUDA' 或 'AMD'

# 模型加载
with open(VISION_PKL_PATH, 'rb') as f:
    vision_run = pickle.load(f)

# 推理
output = vision_run(**inputs).contiguous().realize().uop.base.buffer.numpy()
```

在 x86 开发环境下：
- 默认使用 CPU（`DEV=CPU`）
- 有 NVIDIA GPU 可设置 `DEV=CUDA`
- 有 AMD GPU 可设置 `DEV=AMD`

### 4.4 相机内参

使用 openpilot 标准内参 `DEVICE_CAMERAS[("pc", "unknown")]` = `_ar_ox_config`：

| 相机 | openpilot 焦距 | Carla FOV | Carla 焦距 | 差异 |
|------|---------------|-----------|-----------|------|
| 窄角 (fcam) | 2648.0 | 40° | 2648.6 | 0.02% |
| 广角 (ecam) | 567.0 | 120° | 556.6 | 1.8% |

差异在标定容差范围内。

### 4.5 视觉模型输入名到相机的映射

从模型元数据 `vision_input_shapes` 确认：
- `img` → 主相机（窄角 FOV=40°）→ 使用 MED 模型参数 (bigmodel_frame=False)
- `big_img` → 广角相机（FOV=120°）→ 使用 SBIG 模型参数 (bigmodel_frame=True)

判断依据：`modeld.py:454-455` 中 `'big' in name` → extra (广角)。

## 5. 与现有代码的复用关系

| 现有代码 | 复用方式 |
|----------|---------|
| `selfdrive/modeld/models/commonmodel_pyx.so` | 导入 CLContext, DrivingModelFrame；扩展 prepare_from_yuv |
| `selfdrive/modeld/models/driving_vision_tinygrad.pkl` | 直接加载运行 |
| `selfdrive/modeld/parse_model_outputs.py` | 直接导入 Parser |
| `selfdrive/modeld/constants.py` | 直接导入 ModelConstants, Plan, Meta |
| `selfdrive/locationd/calibrationd.py` | 移植 Calibrator 核心算法（handle_cam_odom 等） |
| `common/transformations/model.py` | 直接导入 get_warp_matrix |
| `common/transformations/camera.py` | 直接导入坐标系、内参、投影函数 |
| `common/transformations/orientation.py` | 直接导入 rot_from_euler, euler_from_rot |
| `tools/sim/rgb_to_nv12.cl` | 通过 pyopencl 加载执行 |
| `tools/sim/bridge/carla/carla_world.py` | 参考 Carla API 用法，重写为独立版本 |

## 6. 实现步骤

| 阶段 | 内容 | 详细说明 | 预估代码量 |
|------|------|---------|-----------|
| 0 | 扩展 commonmodel_pyx | 添加 `prepare_from_yuv()` 方法 + 重新编译 | ~20 行 Cython |
| 1 | `carla_world.py` | Carla 连接、同步模式、车辆/autopilot、双路相机、NPC | ~170 行 |
| 2 | `vision_model.py` | RGB→NV12、OpenCL 预处理、模型加载/推理、输出解析 | ~200 行 |
| 3 | `calibrator.py` | KnownPoseCalibrator + OnlineCalibrator（移植 calibrationd 核心） | ~200 行 |
| 4 | `visualizer.py` | 车道线/路沿/前车绘制、信息面板、视频写入、窗口缩放 | ~200 行 |
| 5 | `run.py` | 主循环、命令行参数、TICKS_PER_FRAME 节拍、信号处理 | ~120 行 |
| 6 | 集成测试 | 先用 KnownPoseCalibrator 验证全链路，再测试 OnlineCalibrator | — |

总计约 **900 行** Python 代码 + 20 行 Cython 扩展。

## 7. 开发优先级

**第一阶段: 最小可运行版本 (KnownPoseCalibrator)**
1. 阶段 0: 扩展 commonmodel_pyx
2. 阶段 1: carla_world.py
3. 阶段 2: vision_model.py
4. 阶段 4: visualizer.py (基础版)
5. 阶段 5: run.py
6. 使用 `--perfect-cam` 或已知角度运行，验证全链路正确性

**第二阶段: 在线标定**
1. 阶段 3: calibrator.py (OnlineCalibrator)
2. 用非零 pitch/yaw 启动，观察标定收敛过程

**第三阶段: 完善**
1. 视频输出
2. 更多可视化信息
3. 性能优化
