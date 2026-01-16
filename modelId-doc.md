**概览**
- `modeld` 是 openpilot 的驾驶模型推理进程，负责从相机图像与车辆/标定状态中生成道路、车道线、障碍物与轨迹等高层感知与规划信号，并通过消息总线发布给后续模块（如规划与控制）。
- 进程入口：`selfdrive/modeld/modeld.py:1`。在整套系统中由进程管理器拉起：`system/manager/process_config.py:39`（条目名 `modeld`）。

**模型组成**
- 驾驶模型拆分为两个子网络：
  - 视觉子模型（Vision）
    - 文件：`selfdrive/modeld/models/driving_vision_tinygrad.pkl`（tinygrad 运行器）、`selfdrive/modeld/models/driving_vision.onnx`（ONNX）、`selfdrive/modeld/models/driving_vision_metadata.pkl`（I/O 元数据）。
    - 功能：对双目（窄/广角）道路图像进行特征提取与即时感知，输出车道线、道路边缘、路面几何、跟车目标、姿态、以及一个时序特征 `hidden_state` 供策略子模型使用。
  - 策略子模型（Policy，temporal policy）
    - 文件：`selfdrive/modeld/models/driving_policy_tinygrad.pkl`、`selfdrive/modeld/models/driving_policy.onnx`、`selfdrive/modeld/models/driving_policy_metadata.pkl`。
    - 功能：在 `hidden_state` 的基础上，结合驾驶意图（desire）、交通习惯（左右道）与时序上下文，预测未来轨迹 `plan`（33 个时刻、位置/速度/姿态等 15 维字段）与 `desire_state`。
- 驾驶员监控模型独立于 `modeld`（进程 `dmonitoringmodeld`），此文不展开。

**输入数据**
- 相机图像（来自 `camerad` 的 VisionIPC）
  - 主流：ROAD 或 WIDE_ROAD，另一路为对侧（用于双摄融合）。
  - `modeld` 通过 `VisionIpcClient` 订阅：`selfdrive/modeld/modeld.py:240` 起。
  - 标定变换：使用设备与相机内参、在线标定欧拉角生成 3x3 透视变换矩阵（warp）：`openpilot.common.transformations.camera.get_warp_matrix`。
  - 图像在 OpenCL 环境下加载/转换为模型输入缓冲，TICI/QCOM 上零拷贝绑定（`CLContext` + `qcom_tensor_from_opencl_address`）。
- 视觉子模型输入张量（两路同构）：
  - 名称：`img`（主摄），`big_img`（广角）。
  - 形状：`(1, 12, 128, 256)`，由两帧拼接而成（每帧 6 通道）。
  - 通道打包（YUV420）：Y 通道按棋盘格拆分为 4 个 128x256 下采样块（`Y[::2,::2]`, `Y[::2,1::2]`, `Y[1::2,::2]`, `Y[1::2,1::2]`），U/V 为 128x256 的上采样块；两帧共 12 通道。具体参见 `selfdrive/modeld/models/README.md:1`。
- 策略子模型输入张量：
  - `features_buffer`：来自视觉输出的 `hidden_state` 经时间队列整形（5 Hz 上下文），形状参见 `driving_policy_metadata.pkl`。
  - `desire_pulse`：驾驶意图脉冲（上升沿触发），由 `DesireHelper` 与上次 desire 的差分生成。
  - `traffic_convention`：左右行驶习惯 one-hot（来自 `driverMonitoringState.isRHD`）。
- 频率与时序：
  - 相机与模型运行频率：`MODEL_RUN_FREQ = 20` Hz。
  - 上下文频率：`MODEL_CONTEXT_FREQ = 5` Hz，用于 `features_buffer` 时序堆叠。
  - 每次视觉推理使用两帧（`N_FRAMES = 2`）。

**推理流程**
1) 设备初始化与运行器装载
   - 选择计算设备（PC 上缺省 `CPU`，TICI 上为 `QCOM`，特殊场景可切换 `AMD`/`USBGPU`）：`selfdrive/modeld/modeld.py:2`–`selfdrive/modeld/modeld.py:10`。
   - 读取视觉与策略子模型的 tinygrad 运行器与 I/O 元数据（输入形状、输出切片）：`selfdrive/modeld/modeld.py:38`–`selfdrive/modeld/modeld.py:41` 与 `ModelState.__init__`。
   - 构建 OpenCL 上下文与帧包装器 `DrivingModelFrame`（图像预处理/几何变换）。
2) 接收帧与同步
   - 订阅主路与广角视频流，基于 SOF/EOF 时间戳对齐两路帧。
   - 管理丢帧：统计 VisionIPC 帧号差异，必要时仅做预处理、跳过一次模型推理以赶上实时（`prepare_only`）。
3) 视觉前向
   - 将主/广角图像写入对应 OpenCL 缓冲并转换为 `(1,12,128,256)` 的 uint8 张量。
   - 执行视觉子模型 tinygrad 前向，得到一维输出 `vision_output`，按 `metadata['output_slices']` 切片并经 `Parser.parse_vision_outputs` 解析为结构化张量。
   - 关键输出：`lane_lines`、`lane_lines_prob`、`road_edges`、`lead/lead_prob`、`pose/road_transform`、以及 `hidden_state`。
4) 准备策略输入并前向
   - 将 `hidden_state` 入队形成 `features_buffer`（5 Hz）；合成 `desire_pulse` 与 `traffic_convention`。
   - 执行策略子模型 tinygrad 前向，解析得到 `plan` 与 `desire_state`。
5) 融合与发布
   - 合并视觉与策略输出，计算动作建议（纵向加速度/横向曲率光滑）：`get_action_from_model`。
   - 通过 `fill_model_msg.py` 写入 `modelV2`、`drivingModelData` 与 `cameraOdometry` 并经 `cereal.messaging` 发布。

**输出数据（核心字段）**
- 视觉输出（`Parser.parse_vision_outputs`）：
  - `lane_lines`: 形状 `(1, 4, 33, 2)`，四条线、33 个前视距离采样点、二维 `(y, z)`；
  - `lane_lines_prob`: 形状与含义见元数据，常用为四条线置信度；
  - `road_edges`: `(1, 2, 33, 2)`；
  - `lead`: 领航车轨迹/状态（含多假设 MHP 支持）；
  - `pose`, `wide_from_device_euler`, `road_transform`: 相机位姿与路面变换；
  - `hidden_state`: 视觉特征缓冲（供策略用）。
- 策略输出（`Parser.parse_policy_outputs`）：
  - `plan`: `(1, 33, 15)`，包含位置、速度、加速度、姿态与姿态变化率等字段（`selfdrive/modeld/constants.py:Plan` 切片定义）。
  - `desire_state`: 驾驶意图分布。
- 发布消息（话题与结构）：
  - `modelV2`：车道线、道路边缘、计划轨迹、Meta 指标（脱手/硬刹等概率）、Leads 等；
  - `drivingModelData`：多项数据镜像与多项式路径系数；
  - `cameraOdometry`：相机位姿与路面变换（含方差）。

**坐标系与投影**
- 模型使用车辆前向坐标（x 前、y 左、z 上）；
- 图像到模型/地面坐标的映射依赖内参 K 与外参（设备到视图）矩阵，运行时由在线标定与相机参数推导（`openpilot.common.transformations.camera`）。

**频率与时序控制**
- `MODEL_RUN_FREQ = 20` Hz：模型推理与帧处理频率；
- `MODEL_CONTEXT_FREQ = 5` Hz：策略上下文频率（影响 `features_buffer` 堆叠）；
- `N_FRAMES = 2`：视觉输入由相邻两帧构成，提升时序稳定性；
- 丢帧处理：基于 VisionIPC 帧号与一阶滤波器评估丢帧比例，必要时跳过一次推理以降低延迟。

**计算设备与优化**
- 设备选择：PC 默认 CPU；TICI 设备使用 QCOM（DSP/OpenCL）；可通过环境变量切换 AMD/USBGPU。
- TICI 上使用 `qcom_tensor_from_opencl_address` 绑定 OpenCL 缓冲到 tinygrad 张量，避免复制开销。

**关键代码/文件索引**
- 进程入口与主循环：`selfdrive/modeld/modeld.py:1`
- 视觉/策略模型加载与运行：`selfdrive/modeld/modeld.py:38`、`selfdrive/modeld/modeld.py:139`
- 模型常量与输出字段切片：`selfdrive/modeld/constants.py:1`
- 输出解析（MDN/Softmax/Sigmoid 与结构化切片）：`selfdrive/modeld/parse_model_outputs.py:1`
- 消息填充与发布：`selfdrive/modeld/fill_model_msg.py:1`
- 相机标定/内外参与投影：`common/transformations/camera.py:1`
- 进程管理配置：`system/manager/process_config.py:1`

**开发与测试建议**
- 离线推理/可视化：`tools/offline_lane_overlay.py:1`（加载视觉子模型，对视频叠加车道线）。
- 端到端仿真/回放：使用 `tools/replay` 系列工具复现模型输出与 UI 叠加（可参考 `tools/replay/lib/ui_helpers.py:1` 的绘制逻辑）。
- 构建与单测：`scons -j$(nproc)` 构建原生/绑定；`pytest -q` 运行测试。根据 `pyproject.toml` 中的默认标记与并行设置进行。

**附注**
- 模型 I/O 形状与切片以相应 `*_metadata.pkl` 为准，版本升级可能调整；
- 视觉与策略模型均提供 ONNX 版本，便于第三方运行器或可视化工具（如 Netron）调试；
- 行驶场景中，`plan` 与动作 `Action` 会进行时间平滑与延迟补偿，以匹配执行器动态。

