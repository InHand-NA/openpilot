# 从原始图像到视觉模型输入：图像预处理与通道打包

本文档面向 openpilot 开发者，系统性说明相机采集/仿真生成的原始图像，如何在进入 Vision 模型前被几何变换、重采样与通道打包处理。文中路径均为仓库内文件路径。

## 处理总览
- 输入源：传感器/ISP 输出的 NV12（YUV420 半平面）帧，经 `camerad` 通过 VisionIPC 发布。
- 几何处理：基于在线标定与相机内参，执行透视变换与缩放至模型分辨率（OpenCL 内核）。
- 通道打包：将变换后的 YUV420 拆分/重排为 6 通道；按时间堆叠两帧组成 12 通道。
- 数据类型：保持 `uint8`（0–255），在设备侧实现零拷贝绑定或一次性复制到运行器张量。
- 形状与频率：每路相机输入张量形状 `(1, 12, 128, 256)`；以 20 Hz 运行，使用相隔 `temporal_skip` 的两帧。

## 数据来源与发布
- 传感器与 AE/AGC 控制
  - 相机经高通 ISP 输出 NV12；`camerad` 负责打开设备、曝光/增益控制、生成 VisionIPC 帧。
  - 关键实现：`system/camerad/cameras/camera_qcom2.cc`
    - 初始化与启动：`camerad_thread()`（队列/事件循环）。
    - 曝光控制：`CameraState::set_camera_exposure()` 基于灰度反馈调节曝光时间与增益。
  - 帧封装（NV12 + OpenCL）：`msgq_repo/msgq/visionipc/visionbuf.h`（`VisionBuf` 含 `width/height/stride/uv_offset` 与 `cl_mem`）。
- 仿真（可选）
  - 仿真引擎通常输出 RGB，需先做 RGB→NV12，再走同样的预处理链路：`tools/sim/rgb_to_nv12.cl`、`tools/sim/lib/camerad.py`。

## 几何与尺寸变换（OpenCL）
- 模型分辨率
  - 驾驶视觉模型目标平面尺寸为 `512x256`（Y 平面），UV 为其一半尺寸；定义见 `selfdrive/modeld/models/commonmodel.h` 中 `MODEL_WIDTH/MODEL_HEIGHT`。
- 透视矩阵（3x3 homography）
  - 使用在线标定欧拉角与相机内参与视图变换构造：`common/transformations/model.py:get_warp_matrix`。
- 透视重采样与缩放
  - OpenCL 内核 `warpPerspective` 对 NV12 的 Y、U、V 分别进行双线性插值重采样；UV 使用按 0.5 尺度缩放后的矩阵。
  - 调度与参数设置：`selfdrive/modeld/transforms/transform.cc`；内核实现：`selfdrive/modeld/transforms/transform.cl`。

## 标准视图定义（模型基准）
- 视图坐标系（view frame）
  - 定义：x→右、y→下、z→前；`common/transformations/camera.py` 中由 `view_frame_from_device_frame` 给出固定变换。
  - 相机内参（K）将 view 坐标投影到像素平面；`CameraConfig.intrinsics` 即 K。
- 模型的标准相机视图（canonical model view）
  - 训练与推理的几何基准采用“零外参（roll/pitch/yaw=0, height=0）的 view 帧 + 预设模型内参（medmodel 或 sbigmodel）”形成的理想针孔相机视图。
  - 代码上以 `medmodel_frame_from_calib_frame` / `sbigmodel_frame_from_calib_frame` 表示，并取其前 3 列求逆得到 `calib_from_model`（模型视图→calib 帧）。
- 当前设备姿态到像素平面的映射
  - 在线标定输出 `device_from_calib_euler`；通过 `view_frame_from_device_frame` 与当前真实相机内参 K，得到 `camera_from_calib = K @ view_frame_from_device_frame @ rot(device_from_calib_euler)`。
- 最终透视矩阵含义
  - `warp_matrix = camera_from_calib @ calib_from_model`，表示“从模型的标准相机视图（canonical model view）到当前设备姿态下真实相机像素平面”的映射。
  - OpenCL 内核以目标（模型尺寸）像素逐点反采样源图像：对输出像素 (dx,dy)，用 `warp_matrix` 计算源像素 (sx,sy) 并做双线性插值。

## 通道打包与时序堆叠
- 单帧 6 通道（YUV420 → 6）
  - Y 通道拆成 4 个棋盘格子通道：等价 `Y[::2,::2]`, `Y[::2,1::2]`, `Y[1::2,::2]`, `Y[1::2,1::2]`，每个尺寸 `(H/2, W/2)`。
  - U、V 作为两个半分辨率通道，尺寸同上。
  - 实现：`selfdrive/modeld/transforms/loadyuv.cc` 调用 `loadys/loaduv`；内核 `selfdrive/modeld/transforms/loadyuv.cl`。
- 两帧堆叠（共 12 通道）
  - 每次前向使用两帧：一帧为历史帧（相隔 `temporal_skip`），一帧为当前帧。
  - 实现：`selfdrive/modeld/models/commonmodel.cc` 中对 20Hz 图像缓冲的移动与复制，组合为输入的“前半帧+后半帧”。
- 输入形状与名称
  - 每路相机输入张量为 `(1, 12, 128, 256)`；主摄对应 `img`，广角对应 `big_img`。详见 `selfdrive/modeld/models/README.md`。

## 数据类型与张量绑定
- 设备（QCOM/TICI）
  - 通过 `qcom_tensor_from_opencl_address` 将 OpenCL 缓冲零拷贝绑定为 tinygrad 张量（`uint8`）。
- 其他平台
  - 从 OpenCL 读取到 `numpy.uint8`，再构造张量。
- 代码入口：`selfdrive/modeld/modeld.py:ModelState.run`。

## 端到端数据流（简述）
1) `camerad` 采集并发布 NV12 帧（VisionIPC）。
2) `modeld` 订阅主/广角流，对齐帧并计算透视矩阵。
3) 使用 OpenCL：`warpPerspective` 生成目标分辨率的 Y/U/V；`loadys/loaduv` 打成 6 通道。
4) 将历史帧与当前帧拼接，得到 12 通道输入张量（每路相机）。
5) 传入视觉子模型进行前向推理。

## 关键参数与频率
- `N_FRAMES=2`：两帧时序输入。
- `MODEL_RUN_FREQ=20` Hz，`MODEL_CONTEXT_FREQ=5` Hz → `temporal_skip=4`。
- 分辨率 `512x256`（Y）；输入张量 `(1, 12, 128, 256)`。
- 定义见：`selfdrive/modeld/constants.py`、`selfdrive/modeld/models/commonmodel.h`。

## 源码索引（快速定位）
- 进程与输入装载：`selfdrive/modeld/modeld.py`
- 透视矩阵：`common/transformations/model.py`
- 透视重采样核：`selfdrive/modeld/transforms/transform.cc`、`selfdrive/modeld/transforms/transform.cl`
- YUV 打包核：`selfdrive/modeld/transforms/loadyuv.cc`、`selfdrive/modeld/transforms/loadyuv.cl`
- VisionIPC 帧结构：`msgq_repo/msgq/visionipc/visionbuf.h`
- 模型输入说明：`selfdrive/modeld/models/README.md`
- 相机与 AE/AGC：`system/camerad/cameras/camera_qcom2.cc`

## 备注
- 视觉模型输入保持 `uint8`，无额外均值/方差归一化；归一化与颜色映射学习在模型首层完成。
- 双摄场景（主路/广角）会分别构建一套 `(1,12,128,256)` 输入并在模型中融合。
- 仿真环境下仅在 RGB→NV12 处有差异，其余处理一致。
