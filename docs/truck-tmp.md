# 卡车前视ADAS模型端到端方案

## 最终目标

Openpilot预训练的driving_vision.onnx模型在相机安装高度超过1.8m之后准确度显著降低，不适用于卡车安装场景。

基于Openpilot预训练的driving_vision.onnx模型，微调得到新模型，适用于小轿车和卡车等多种场景。实现LDW和FCW业务功能。

## 阶段性目标

### 1. 预训练模型重构（解析、复刻、裁减、验证）

- export_pretrain_model.py:
- carla_labeling.py:
- view_data.py:
- model_eval.py:


### 2. 微调pt模型

从carla系统中采集1.8m ~ 3.0m安装高度的训练数据，对pt模型进行微调。

#### 2.1 训练数据构建方案
#### 2.2 训练方案
#### 2.3 验证

### 3. 部署微调模型
