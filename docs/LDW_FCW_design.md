# 基于 openpilot 的 Dashcam 视频 LDW/FCW 技术方案（含实现要点）

本文面向“仅有一路 Dashcam 视频（HFOV≈120°，已知相机内参）”的离线处理场景，目标是在视频上完成：
- 车道线检测与拟合：输出道路坐标系下的车道线曲线，同时给出 2D 像素坐标系下的线段/采样点；
- LDW（Lane Departure Warning）：结合车身尺寸与车辆姿态，判断是否发生车道偏离并告警；
- 前车（Lead）运动信息估计与 FCW（Forward Collision Warning）：估计相对距离/速度/加速度，判断碰撞风险。

方案充分参考并复用 openpilot 的坐标系定义、在线标定流程、模型输出结构与控制侧判定逻辑，配合离线视频处理所需的工程化改造，给出一套可落地的实现路径。


**目录**
- 背景与可复用模块
- 坐标系、相机模型与标定
- 视频预处理（去畸变与几何归一化）
- 车道线检测与拟合
- LDW 判定逻辑
- 前车（Lead）检测、跟踪与尺度/运动估计
- FCW 判定逻辑
- 数据结构与接口建议（与 openpilot 对齐）
- 开发与测试计划
- 风险与改进方向


## 背景与可复用模块
openpilot 已在“设备+车辆传感器”条件下实现了一整套从视频到规划控制的链路，其中与本文密切相关的实现与消息定义：
- 模型与输出封装
  - 视觉输出的封装：`selfdrive/modeld/fill_model_msg.py`
    - 车道线：`modelV2.laneLines[0..3]` 与 `laneLineProbs/Std`
    - 路沿：`modelV2.roadEdges[0..1]`
    - 前车：`modelV2.leadsV3[0..2]`，含 `x,y,v,a` 的时序轨迹与概率
  - 相机里程计与路面变换：`fill_pose_msg()` 发布 `cameraOdometry`，包含 `trans/rot` 与 `roadTransformTrans`
- 在线标定（外参）：`selfdrive/locationd/calibrationd.py`
  - 融合 `cameraOdometry` 与车身状态，在“直且稳”的片段渐进估计 `rpyCalib`（滚转/俯仰/偏航）与相机高度 `height`
  - 提供稳定性判断与持久化写入 `Params`
- 坐标与投影工具：
  - 相机/设备/路面坐标系定义与工具：`common/transformations/camera.py`
  - 模型输入几何与单应矩阵：`common/transformations/model.py` 中 `get_warp_matrix()`
- 纵向融合与 Lead 逻辑：`selfdrive/controls/radard.py`
  - 将视觉 Leads 与雷达轨迹融合；无雷达时使用视觉 Leads 回退（`get_RadarState_from_vision`）
- LDW 判定参考：`selfdrive/controls/lib/ldw.py`

尽管我们只有离线视频（无车速/IMU/雷达），上述模块提供了关键“算法形态与接口设计”的参考：
1) 坐标系定义与外参参数化（rpy + height）；
2) 将像素点投影到“道路坐标系”的几何路径；
3) 车道线/前车的消息结构与概率化表述；
4) FCW/LDW 的阈值与时序门控思路。


## 坐标系、相机模型与标定

### 坐标系约定（与 openpilot 对齐）
- 设备坐标系（device/mesh）：x 前、y 右、z 下；
- 视图坐标系（view）：x 右、y 下、z 前；
- 道路坐标系（road）：x 前、y 左、z 上。

典型转换见：`common/transformations/camera.py`
- `view_frame_from_device_frame` 固定旋转（device->view）；
- `get_view_frame_from_road_frame(roll, pitch, yaw, height)` 给出路面->视图的 3x4 外参。

### 相机内参与畸变
- 已知 HFOV≈120° , 相机内参矩阵 `K`（3x3）和针孔模型畸变参数。

### 外参（姿态与高度）离线估计
由于离线视频缺乏 IMU/车速，参考 `calibrationd.py` 的“直且稳”思想，采用纯视觉估计：
1) 初值：
   - 俯仰与偏航初始值设置为0；
   - 相机高度 h 取车辆安装经验值（例如 1.2m），后续自适应微调。
2) 时序稳健：
   - 连续帧上提取路面/车道几何，使用滑动窗口线性衰减平均（参见 `moving_avg_with_linear_decay` 思想），
     对 rpy 与 h 逐步收敛；
3) 条件门控：
   - 仅在“场景稳定”时更新（小俯仰/小偏航变化、画面平稳），避免坡道/过弯干扰；
4) 异常回退：
   - 若短时间内姿态散布过大（参考 `MAX_ALLOWED_PITCH_SPREAD/YAW_SPREAD`），回退到上一稳定块并重启估计。

输出外参参数化为 `roll≈0, pitch, yaw, height`，与 openpilot 的 `liveCalibration.rpyCalib/height` 一致。


## 视频预处理（去畸变与几何归一化）
目标：将原始帧统一映射到“几何归一化”的模型平面，以便后续算法稳定工作。

参考 `common/transformations/model.py`：
- 构造单应矩阵 `warp = get_warp_matrix(device_from_calib_euler, intrinsics)`；
- 其中 `device_from_calib_euler = [rx, ry, rz]` 对应 `roll, pitch, yaw`（右手系），`intrinsics` 为目标相机 K；
- 将原始图像按 `warp` 做投影变换，得到“姿态归一化”的图像帧；
- 在估计出外参与相机高度后，可进一步构建路面坐标系到“姿态归一化”的图像坐标系的变换矩阵，用于后续把道路坐标系的数据变换到像素图像上；

工程建议：
- 去畸变 -> 姿态归一化（warp 到标准视角）；
- 帧率与时序：若后续检测/跟踪以固定频率运行，可做滑窗缓存与亚采样（见 `ModelState.InputQueues` 的设计思想）。


## 模型：车道线检测与拟合

使用Openpilot的视觉网络，从视频数据中回归车道线、相机里程计和前车Lead数据。


### 坐标与投影（像素↔道路）
已知内参 K 与外参（R、h），将像素点 p = [u,v,1]^T 投影到道路坐标（假定平整地面 z=0）：
1) 归一化视线：`r_cam = K^{-1} p`；
2) 转到道路系：`d_r = R_cr · r_cam`，其中 `R_cr = (R_rc)^T`，R_rc 为道路->相机旋转；
3) 相机在道路系位置 `C = (0, 0, h)`；射线方程 `P(s) = C + s·d_r`；
4) 与地面 z=0 相交：`s* = -h / d_r.z`，则 `X = s*·d_r.x`，`Y = s*·d_r.y`。

实现可直接复用 `common/transformations/camera.py` 与 `model.py` 提供的基元（`normalize/denormalize`、各类外参矩阵），避免重复推导与符号误差。

### 输出形式
- 像素系：输出每条车道在原图上的采样点序列或线段集；
- 道路系：以 x 前向（米）为自变量，给出 `y(x)` 或三次样条系数；
- 置信度：根据边缘强度、连续性与拟合残差，产出 `leftProb/rightProb` 类似的概率，用于 LDW 门控。


## LDW 判定逻辑

TODO


## 前车（Lead）检测、跟踪与尺度/运动估计
使用Openpilot的视觉网络输出的模型，得到前车Lead的概率、位置、速度、加速度等信息，用于后续FCW规则。


## 风险与改进方向
- 外参稳定：仅用视频估计 rpy/h 对坡道/起伏敏感，建议在 BEV 下引入地面线/标志点稳健估计与时序滤波；
- 畸变影响：120° 近场形变明显，几何法估距在近距离会有系统误差，需要畸变模型或数据驱动校正；
- 目标检测鲁棒性：夜晚/雨雾/逆光下模型置信度下降，宜引入时序先验与多目标融合；
- 评估与标注：建议构建最小标注集（车道/前车距离）用于阈值回归与回归测试。


——
参考实现位置（阅读索引）：
- `selfdrive/modeld/fill_model_msg.py`：车道/路沿/Lead 与 FCW 相关封装
- `selfdrive/locationd/calibrationd.py`：在线外参估计算法与稳定性策略
- `common/transformations/camera.py`：坐标系定义、内外参工具
- `common/transformations/model.py`：模型平面几何、单应矩阵 `get_warp_matrix`
- `selfdrive/controls/lib/ldw.py`：LDW 判定逻辑与阈值参考
- `selfdrive/controls/radard.py`：Lead 融合/跟踪与结构定义（无雷达时的视觉回退）

本文档作为后续开发的技术参考与“接口契约”，建议实现过程中严格对齐上述坐标/数据结构，便于后续与 openpilot 生态复用与联调。

