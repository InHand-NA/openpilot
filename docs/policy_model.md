# 策略模型（Temporal Policy）输入与输出说明

本篇聚焦 openpilot 驾驶模型中的“策略模型”（temporal policy）的数据接口，说明它在整体架构中的位置，以及其主要输入、输出、形状与典型取样频率，便于开发与调试。

## 概览
- 策略模型与视觉模型共同构成“驾驶模型”。视觉模型从前置相机图像中提取即时感知与时序特征；策略模型在此基础上结合驾驶意图与规则，输出未来的驾驶计划与意图分布。
- 进程实现参见 `selfdrive/modeld/modeld.py:1`。策略与视觉模型分别以元数据（`*_metadata.pkl`）声明 I/O 形状与切片，并在运行时由 `modeld` 装载。
- 下文列出的形状以当前模型元数据与实现为准；不同模型版本可能略有差异（以对应的 `*_metadata.pkl` 为权威）。

## 模型输入（Policy Inputs）
- 关键输入来自三部分：视觉模型输出的中间特征、外部驾驶意图/规则、与可选的控制相关量。

- `features_buffer`（时序特征缓冲）
  - 来源：视觉模型隐藏状态（hidden_state），由 `modeld` 组装成固定长度的时间上下文。
  - 典型形状：`(1, 100, 512)` 表示最近 5s（20 Hz）× 每步 512 维特征；当前实现用 `InputQueues` 在 20 Hz 环境帧率与策略模型训练频率之间做重采样与对齐，具体以 `driving_policy_metadata.pkl` 为准。
  - 相关代码：`selfdrive/modeld/modeld.py:223`、`selfdrive/modeld/modeld.py:286`。

- `desire_pulse`（驾驶意图脉冲）
  - 含义：将外部意图（如直行、变道左/右、变道准备等）编码为 one-hot，并在“上升沿”仅给出一个脉冲，避免策略网络内部状态累积偏移。
  - 典型形状：`(1, 100, 8)`，长度与采样率与上游一致；当前实现会自动从 `DesireHelper` 的结果生成脉冲序列。
  - 相关代码：`selfdrive/modeld/modeld.py:257`–`selfdrive/modeld/modeld.py:263`、`selfdrive/modeld/modeld.py:286`。

- `traffic_convention`（交通习惯）
  - 含义：左右侧通行规则的 one-hot 向量，通常由 `driverMonitoringState.isRHD` 决定。
  - 形状：`(1, 2)`。
  - 相关代码：`selfdrive/modeld/modeld.py:433`–`selfdrive/modeld/modeld.py:435`、`selfdrive/modeld/modeld.py:289`。

- 可选输入（视模型版本而定）
  - `lateral_control_params`：横向控制相关参数，如速度与转向延迟，形状 `2`；帮助策略模型更好地对齐执行器动态。
  - `previous_desired_curvatures`：历史期望曲率序列，形状 `100 × 1`；为曲率预测提供先验与平滑参考。
  - 注：是否启用取决于具体策略模型的 `*_metadata.pkl` 与 `modeld` 中的赋值逻辑；若未显式赋值，默认零值输入。

## 模型输出（Policy Outputs）
- 核心输出包含未来轨迹计划 `plan` 与时刻无关的意图分布 `desire_state`；在解析阶段使用 MDN/Softmax 等函数解码到结构化张量。

- `plan`（未来轨迹与姿态）
  - 形状：`(1, 33, 15)`，表示 33 个时间采样点（非线性时间索引 `T_IDXS`）× 每点 15 维。
  - 字段切片定义见 `selfdrive/modeld/constants.py:Plan`：
    - `POSITION`: 位置 `[x, y, z]`
    - `VELOCITY`: 速度分量
    - `ACCELERATION`: 加速度分量
    - `T_FROM_CURRENT_EULER`: 由当前姿态起的欧拉角增量
    - `ORIENTATION_RATE`: 姿态变化率
  - 不确定性：解析时会输出 `plan_stds`；若模型为多假设（MHP），还会提供 `plan_hypotheses` 与 `plan_weights`（取最大权重假设作为主输出）。
  - 下游使用：`fill_model_msg.py` 将 `plan` 映射到 `modelV2.position/velocity/acceleration/orientation/orientationRate`，并由 `get_action_from_model` 计算期望加速度与曲率。
  - 相关代码：`selfdrive/modeld/parse_model_outputs.py:46`–`selfdrive/modeld/parse_model_outputs.py:66`、`selfdrive/modeld/fill_model_msg.py:33`–`selfdrive/modeld/fill_model_msg.py:62`、`selfdrive/modeld/modeld.py:78`–`selfdrive/modeld/modeld.py:103`。

- `desire_state`（当前意图分布）
  - 含义：对一组预定义意图的概率分布（如直行、变道左、变道右、变道准备等），Softmax 输出。
  - 形状：`(1, 8)`（以 `DESIRE_PRED_WIDTH` 为准）。
  - 下游使用：用于更新 `laneChangeState`、`laneChangeDirection` 等，并可被 UI 与控制参考。
  - 相关代码：`selfdrive/modeld/parse_model_outputs.py:66`、`selfdrive/modeld/fill_model_msg.py:82`、`selfdrive/modeld/modeld.py:312`–`selfdrive/modeld/modeld.py:321`。

## 取样与时序
- 模型运行频率：`MODEL_RUN_FREQ = 20` Hz。
- 上下文频率（训练帧率）：`MODEL_CONTEXT_FREQ = 5` Hz（部分时序输入在内部重采样对齐）。
- 时间索引：`T_IDXS` 共 33 个非线性采样点（近处更密集），详见 `selfdrive/modeld/constants.py:7`–`selfdrive/modeld/constants.py:13`。

## 与消息/模块的映射
- 发布的话题包括：`modelV2`、`drivingModelData`、`cameraOdometry`，字段由 `selfdrive/modeld/fill_model_msg.py:1` 负责填充。
- 控制层直接使用 `modelV2.action` 或从 `plan` 计算的期望加速度/曲率（含平滑与延迟补偿），详见 `selfdrive/modeld/modeld.py:78`–`selfdrive/modeld/modeld.py:103` 与 `openpilot.selfdrive.controls.lib.drive_helpers`。

## 参考
- 代码入口与主循环：`selfdrive/modeld/modeld.py:1`
- 输出解析：`selfdrive/modeld/parse_model_outputs.py:1`
- 常量与切片：`selfdrive/modeld/constants.py:1`
- 模型 I/O 元数据：`selfdrive/modeld/models/driving_policy_metadata.pkl`
- 视觉/策略输入详情：`selfdrive/modeld/models/README.md:1`

