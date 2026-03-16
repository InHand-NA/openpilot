# 隐式方案开发与训练任务计划

> 基于 [training_evaluation_methodology.md](training_evaluation_methodology.md) 第一阶段（隐式方案），
> 对 `checkpoints/inadas_original.pt` 进行多高度联合微调，使模型自适应 1.0m–3.0m 相机安装高度。

---

## 当前状态

| 资产 | 状态 | 路径 |
|------|------|------|
| 预训练模型 (.pt) | ✅ 已就绪 | `checkpoints/inadas_original.pt` |
| 预训练模型 (.onnx) | ✅ 已就绪 | `checkpoints/inadas_original.onnx` |
| tinygrad 推理模型 | ✅ 已就绪 | `checkpoints/inadas_original_tinygrad_cuda.pkl` |
| 多高度原始数据 | ✅ 已采集（163 sessions × 6 heights） | `data/multi_height-0312/` |
| 离线标注 | ✅ 已就绪 | `data/multi_height-0312/Town*/annotations/` |
| 数据清洗脚本 | ❌ 未开发 | — |
| 训练脚本 | ❌ 需重新开发 | — |
| 评价脚本 | ❌ 未开发 | — |

---

## 任务总览

```
Phase A: 数据流水线（标注 → 清洗 → 缓存）
  T1  批量标注
  T2  数据清洗脚本
  T3  数据集划分与 H1 退化验证集
  T4  预处理缓存

Phase B: 训练基础设施
  T5  模型加载与层级分析
  T6  Dataset 与 DataLoader
  T7  损失函数
  T8  训练主脚本

Phase C: 评价基础设施
  T9  离线评价脚本

Phase D: 训练执行与迭代
  T10 快速验证实验
  T11 正式训练实验
  T12 评价与决策

Phase E: 部署
  T13 导出与编译
```

---

## Phase A: 数据流水线

### T1 — 批量标注

**目标**：对已采集的 163 sessions 执行离线标注，生成 H1–H6 各高度的 JSON 标注文件。

**输入**：`data/multi_height-0312/Town*/{H1..H6}/road_*.png, wide_*.png`

**操作**：
```bash
DEV=CUDA python tools/dashcam/annotate_batch.py data/multi_height-0312/ \
    --onnx selfdrive/modeld/models/driving_vision.onnx \
    --min-ll-prob 0.1
```

**输出**：`data/multi_height-0312/Town*/annotations/{H1..H6}/*.json`

**验证**：
- 标注完成的 session 数 = 163
- 运行 `viz/label_stats.py` 检查 pass_rate > 70%

**预估耗时**：2–4 小时（GPU 推理）

**依赖**：无

---

### T2 — 数据清洗脚本

**目标**：新建 `tools/dashcam/train/clean_data.py`，按 training_evaluation_methodology.md §2.1 规则清洗标注数据，输出独立的清洗后数据集。

**文件**：`tools/dashcam/train/clean_data.py`（新建）

**清洗规则**（按顺序执行）：

1. **时序抽帧**：每 5 帧取 1 帧（等效 1 FPS）
   - 采集 `save_every=4` → 保存帧号为 0, 4, 8, 12, 16, 20, ...
   - 抽帧后保留帧号为 0, 20, 40, 60, ...（每隔 5 个保存帧取 1 个）

2. **最低置信度过滤**：仅排除完全无车道线的帧——L-inn prob ≤ 0.05 **且** R-inn prob ≤ 0.05 时丢弃（即至少一条内侧线置信度 > 0.05 即保留）

3. **PRE 帧定位**：为每个被选中帧找到其 PRE 帧（前第 `save_every` 个帧，即帧号差 4），记录到输出中

4. 考虑到H1~H6是同步采集和标注的，数据清洗只需要针对H1做分析即可，H2～H6拥有相同的清洗结果。

> **注意**：lead_prob 不在数据清洗阶段做二值化，保留教师模型的原始软概率。lead 位置损失的 mask 策略在 T7 损失函数中通过运行时阈值实现，详见 T7 第 4 点。
> 这一决策基于对标注数据的统计分析（见 `tools/dashcam/analysis/lead_prob_distribution.py`）：lead_prob 呈双峰分布（52.6% < 0.05, 16.3% ≥ 0.9），中间区 [0.2, 0.8) 占 10.3%，保留这部分的连续置信度信息有利于知识蒸馏。

**输入**：
- 标注目录：`<dataset_dir>/Town*/annotations/{H1..H6}/*.json`
- 图像目录：`<dataset_dir>/Town*/{H1..H6}/road_*.png, wide_*.png`
- clip_info.json：获取 pitch/yaw

**输出**：

1. 在 `<dataset_dir>` 目录下新增 `clean_log.txt`，每行记录一个被选中的帧索引，格式为 `<session_name>/<frame_number>`：
   ```
   Town04_route01/000000
   Town04_route01/000020
   Town04_route01/000040
   ...
   ```
   该索引基于 H1 分析结果生成，H2~H6 共享相同的帧选中结果。

2. 创建 `<dataset_dir>/Town*/cleaned_annotations/{H1..H6}/` 目录，保持与原始 `annotations/` 相同的子目录结构，仅复制被选中帧的标注 JSON（内容与原始标注一致，不做任何数值修改）。

**命令行接口**：
```bash
python tools/dashcam/train/clean_data.py \
    data/multi_height-0312/ \
    --subsample 5 \
    --min-ll-prob 0.05 \
    --dry-run  # 仅统计，不写入
```

**验证工具**：`tools/dashcam/train/verify_clean.py`（新建）

```bash
python tools/dashcam/train/verify_clean.py data/multi_height-0312/
```

自动检查项（全部 PASS 才通过）：

| # | 检查项 | 方法 |
|---|--------|------|
| 1 | 抽帧正确性 | clean_log.txt 中每个帧号 % 20 == 0（save_every=4，subsample=5 → 步长 20） |
| 2 | 置信度过滤 | 对 clean_log.txt 中每帧，读取 H1 原始标注，验证 L-inn prob > 0.05 **或** R-inn prob > 0.05（至少一条内侧线可见） |
| 3 | lead_prob 保留原值 | 遍历所有 cleaned_annotations JSON，lead_prob 值与原始 annotations 中完全一致（未被修改） |
| 4 | PRE 帧存在性 | 对每个被选中帧 N，验证原始图像中帧 N-4 存在（road_*.png 和 wide_*.png） |
| 5 | H1~H6 同步 | 每个 session 的 cleaned_annotations/ 下 H1~H6 拥有完全相同的文件集合 |
| 6 | 完整性 | clean_log.txt 行数 = 任一 session cleaned_annotations/H1/ 下的文件数之和 |

输出示例：
```
[PASS] 抽帧正确性: 2847/2847 帧号均为 20 的倍数
[PASS] 置信度过滤: 2847/2847 帧至少一条内侧线 prob>0.05
[PASS] lead_prob 保留原值: 17082 个 JSON 中 lead_prob 与原始标注一致
[PASS] PRE 帧存在性: 2847/2847 帧均有对应 PRE 帧
[PASS] H1~H6 同步: 163 sessions 全部一致
[PASS] 完整性: clean_log 2847 行 = cleaned_annotations H1 文件数 2847
===== 6/6 PASS =====
```

**依赖**：T1

---

### 数据集划分
#### T3.1 训练数据集划分

**目标**：将 T2 清洗后的数据帧按 8:1:1 全局随机划分为 train/val/test 三个子集。

**文件**：`tools/dashcam/train/split_dataset.py`（新建）

**划分流程**：

```
Step 1: 读取 <dataset_dir>/clean_log.txt，获取所有被选中的帧索引列表
Step 2: 全局随机 shuffle（固定 seed 保证可复现）
Step 3: 按 8:1:1 比例划分 → train / val / test
Step 4: 在 <dataset_dir> 目录下生成 train.txt, val.txt, test.txt，
        每行格式同 clean_log.txt: <session_name>/<frame_number>
```

> 注：已有 1 FPS 时序抽帧（T2 规则 1）防止时序泄漏，因此帧级别 shuffle 是安全的。

**命令行接口**：
```bash
python tools/dashcam/train/split_dataset.py \
    data/multi_height-0312/ \
    --ratio 0.8 0.1 0.1 \
    --seed 42
```

**输出**：
```
data/multi_height-0312/
  train.txt     # 训练集帧索引
  val.txt       # 验证集帧索引
  test.txt      # 测试集帧索引
  split_info.json  # 统计信息：各子集帧数、session 覆盖率等
```

**验证**：脚本运行结束时自动执行以下检查，全部 PASS 才正常退出：

| # | 检查项 | 方法 |
|---|--------|------|
| 1 | 总帧数守恒 | len(train) + len(val) + len(test) == len(clean_log.txt) |
| 2 | 无交集 | train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅ |
| 3 | 比例偏差 | 实际比例与目标比例的绝对偏差 < 1%（因整数取整） |
| 4 | Session 覆盖率 | 每个子集覆盖的 session 数 / 总 session 数 > 80% |
| 5 | 索引合法性 | 每行均可解析为 `<session_name>/<frame_number>`，且对应的 cleaned_annotations 文件存在 |

检查结果写入 split_info.json 的 `"verification"` 字段。

**依赖**：T2

#### T3.2 H1 退化验证集

**目标**：从 T2 抽帧规则**未选中**的 H1 帧中抽选退化验证集，用于训练过程中监测 H1 性能是否退化。

**文件**：`tools/dashcam/train/h1_holdout_split.py`（新建）

**划分流程**（严格按以下顺序）：

```
Step 1: 遍历所有 session 的 H1 标注，列出全部帧号
Step 2: 排除 clean_log.txt 中已选中的帧（这些帧已用于 T3.1 的 train/val/test）
Step 3: 对剩余帧应用 T2 相同的最低置信度过滤（L-inn prob > 0.05 或 R-inn prob > 0.05）
Step 4: 从通过过滤的帧中随机抽出 10% → H1 退化验证集
Step 5: 生成索引文件
```

**输出**：
```
data/multi_height-0312/
  h1_holdout.txt   # 每行: <session_name>/<frame_number>（仅 H1 帧）
```

**命令行接口**：
```bash
python tools/dashcam/train/h1_holdout_split.py \
    data/multi_height-0312/ \
    --clean-log data/multi_height-0312/clean_log.txt \
    --holdout-ratio 0.1 \
    --min-ll-prob 0.05 \
    --seed 42
```

**验证**：脚本运行结束时自动执行以下检查，全部 PASS 才正常退出：

| # | 检查项 | 方法 |
|---|--------|------|
| 1 | 与训练集无交集 | h1_holdout ∩ clean_log = ∅ |
| 2 | 比例正确 | holdout 帧数 / 候选帧总数 与 --holdout-ratio 的偏差 < 2% |
| 3 | 置信度过滤 | 每帧读取 H1 原始标注，验证 L-inn prob > 0.05 或 R-inn prob > 0.05 |
| 4 | Session 覆盖率 | holdout 覆盖的 session 数 / 总 session 数 > 80% |
| 5 | 索引合法性 | 每行对应的 H1 标注文件存在 |

检查结果打印到 stdout 并记录到 `h1_holdout_info.json`。

**依赖**：T2

#### T3.3 Mini数据集

从T3.1和T3.2的成果中摘取出一份很小的数据集子集，用于调试训练脚本。在这些数据集上不关注模型精度。

---

### T4 — 预处理缓存

**目标**：新建 `tools/dashcam/train/build_cache.py`，将划分后的数据预处理为训练可直接加载的 npz 缓存。

**文件**：`tools/dashcam/train/build_cache.py`（新建）

**处理流程**（对每个 数据帧 引用）：
1. 读取 curr/prev 的 road + wide 原始 PNG
2. 从 `pitch_deg`, `yaw_deg` 计算 warp 矩阵
3. GPU OpenCL 预处理：BGR → NV12 → warp → loadyuv → `(6, 128, 256)` uint8
4. 拼接 prev + curr → `(12, 128, 256)` uint8（窄焦和广角各一份）
5. 与标注张量一起写入缓存 npz

**输出**：
```
data/caches/
  train/   *.npz   # 每个 npz: road_input(12,128,256), wide_input(12,128,256), labels...
  val/     *.npz
  test/    *.npz
  h1_holdout/ *.npz
```

npz文件名格式为"\<scene name\>-H\<k\>-\<xxxxxx\>.npz"

**每个缓存 npz 包含**：

| 字段 | 形状 | 类型 |
|------|------|------|
| `road_input` | `(12, 128, 256)` | uint8 |
| `wide_input` | `(12, 128, 256)` | uint8 |
| `camera_height` | scalar | float32 |
| `lane_lines` | `(4, 33, 3)` | float32 |
| `lane_lines_prob` | `(4,)` | float32 |
| `road_edges` | `(2, 33, 3)` | float32 |
| `road_edges_prob` | `(2,)` | float32 |
| `lead` | `(3, 6, 4)` | float32 |
| `lead_prob` | `(3,)` | float32 |
| `pose` | `(6,)` | float32 |
| `road_transform` | `(6,)` | float32 |
| `wide_from_device_euler` | `(3,)` | float32 |

**命令行接口**：
```bash
python tools/dashcam/train/build_cache.py \
    data/multi_height-0312/ \
    --output data/caches/ \
    --gpu  # 使用 OpenCL GPU 预处理
```

脚本自动读取 `<dataset_dir>` 下的 `train.txt`, `val.txt`, `test.txt`, `h1_holdout.txt` 索引文件，
结合 `cleaned_annotations/` 中的标注 JSON 和原始图像，生成各子目录的 npz 缓存。
h1_holdout 中的帧仅处理 H1 高度。

**验证工具**：`tools/dashcam/train/verify_cache.py`（新建）

```bash
python tools/dashcam/train/verify_cache.py \
    data/caches/ \
    --dataset-dir data/multi_height-0312/ \
    --visual-samples 5  # 生成可视化对比图
```

自动检查项（全部 PASS 才通过）：

| # | 检查项 | 方法 |
|---|--------|------|
| 1 | 帧数一致性 | 各子目录 npz 文件数与 train.txt/val.txt/test.txt/h1_holdout.txt 中的帧数 × 高度数一致 |
| 2 | npz 字段完整性 | 随机抽取 20 个 npz，检查所有必需字段存在、shape 和 dtype 正确 |
| 3 | 数值范围 | road_input/wide_input ∈ [0,255]；camera_height ∈ [0.8, 3.5]；lane_lines_prob ∈ [0,1]；lead_prob ∈ [0, 1] |
| 4 | H1 holdout 纯度 | h1_holdout/ 下所有 npz 的文件名包含 `-H1-`，camera_height ≈ 1.22m（±0.1） |
| 5 | 图像一致性 | 随机抽样 N 帧，独立执行 PNG→warp→loadyuv 流水线，与 npz 中 road_input/wide_input 做逐像素对比，MAE=0 |
| 6 | 标注一致性 | 随机抽样 N 帧，读取 cleaned_annotations JSON，与 npz 中标注字段逐值对比，误差 < 1e-6 |

`--visual-samples N` 时额外生成可视化 PNG：
```
data/caches/verify_samples/
  sample_001_road.png   # 左=原始 PNG warp 后，右=npz 中 road_input 还原
  sample_001_wide.png
  ...
```

输出示例：
```
[PASS] 帧数一致性: train=13686, val=1710, test=1710, h1_holdout=1205
[PASS] npz 字段完整性: 20/20 个样本 shape/dtype 全部正确
[PASS] 数值范围: 20/20 个样本全部在合法范围内
[PASS] H1 holdout 纯度: 1205/1205 文件均为 H1, height=1.22±0.02
[PASS] 图像一致性: 5/5 帧 MAE=0.0（逐像素完全一致）
[PASS] 标注一致性: 5/5 帧最大误差=0.0
===== 6/6 PASS =====
可视化已保存到 data/caches/verify_samples/
```

**预估耗时**：~30 分钟（GPU 预处理）

**依赖**：T3

---

## Phase B: 训练基础设施

### T5 — 模型加载与层级分析

**目标**：编写模型加载工具，从 `checkpoints/inadas_original.pt` 加载预训练权重到 PyTorch 模型，并分析模型层级结构以支持渐进解冻。

**文件**：`tools/dashcam/train/model_loader.py`（新建）

**核心功能**：

1. **加载预训练权重**：
   - 使用 onnx2torch 从 `inadas_original.pt` 加载 state_dict
   - 映射到可训练的 PyTorch nn.Module

2. **层级结构分析**：
   - 打印模型各层名称、参数量、shape
   - 识别并标记：stem / stage0 / stage1 / stage2 / stage3 / fc / output_heads
   - 输出层级分组字典，供训练脚本使用

3. **冻结/解冻接口**：
   ```python
   def get_param_groups(model, phase: int) -> list[dict]:
       """根据 phase (1-4) 返回对应的 optimizer param_groups"""

   def set_phase(model, phase: int):
       """冻结/解冻对应层，设置 requires_grad"""
   ```

4. **前向验证**：加载后用随机输入验证输出 shape 为 977 维

**验证**：
- 加载无报错
- 各层参数量之和 ≈ 23M
- 随机输入的输出 shape = `(1, 977)`
- phase 1/2/3/4 的可训练参数量依次递增

---

**预训练模型 H1 精度验证**：`tools/dashcam/train/eval_pretrained_h1.py`（新建）

用 `inadas_original.pt` 在 H1 标注数据上做推理，建立预训练基线指标。该基线有两个用途：
1. **sanity check**：验证 pt 权重加载正确（推理结果应与 ONNX 标注高度一致，因为标注本身由同一模型产生）
2. **退化基线**：训练后对比 H1 精度退化程度时的参照值

```bash
python tools/dashcam/train/eval_pretrained_h1.py \
    --checkpoint checkpoints/inadas_original.pt \
    --cache-dir data/caches/h1_holdout/ \
    --output checkpoints/eval_pretrained_h1.json
```

**评估内容**：

| 指标 | 计算方法 | 预期结果 |
|------|---------|---------|
| lane_lines y_lat MAE | 预测均值 vs H1 标注，仅 prob>0.5 的车道线 | **极低**（< 0.05m），因为标注来自同一模型 |
| lane_lines z_height MAE | 同上 | **极低** |
| lane_lines_prob MAE | 预测 prob vs 标注 prob | **极低**（< 0.01） |
| lead x_fwd MAE | 预测 vs 标注，仅 lead_prob=1.0 的帧 | **极低** |
| lead_prob accuracy | 预测二值化 vs 标注二值化 | **极高**（> 99%） |
| pose MAE | 各分量 MAE | **极低** |
| road_transform MAE | 各分量 MAE | **极低** |
| total_loss（用 T7 损失函数） | 与训练时同一损失函数计算 | 记录数值，作为 epoch 0 的 H1 baseline |

**判定标准**：
- 所有位置 MAE < 0.1m → PASS（证明 pt 加载正确）
- 若 MAE 明显偏大 → 模型加载或预处理流水线有 bug，需排查

**输出**：`eval_pretrained_h1.json`，包含各指标数值，供 T8 训练脚本读取作为 `val_loss_h1_baseline`。

**依赖**：无（仅依赖 checkpoint 文件）；运行时依赖 T4 生成的 h1_holdout 缓存和 T7 损失函数

---

### T6 — Dataset 与 DataLoader

**目标**：新建训练用 Dataset 类，从缓存 npz 加载数据。

**文件**：`tools/dashcam/train/height_dataset.py`（新建）

**Dataset 类**：

```python
class MultiHeightDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, augment: bool = False):
        """
        cache_dir: 包含 *.npz 缓存文件的目录
        augment: 是否启用数据增强（亮度/对比度抖动）
        """

    def __getitem__(self, idx):
        """
        返回:
          road_input: (12, 128, 256) uint8 tensor
          wide_input: (12, 128, 256) uint8 tensor
          targets: dict {
            lane_lines: (4, 33, 2) float32  # y_lat, z_height（不含 x 维度，x 是固定的 X_IDXS）
            lane_lines_prob: (4,) float32
            road_edges: (2, 33, 2) float32
            road_edges_prob: (2,) float32
            lead: (3, 6, 4) float32
            lead_prob: (3,) float32
            pose: (6,) float32
            road_transform: (6,) float32
            wide_from_device_euler: (3,) float32
            camera_height: scalar float32
          }
        """
```

**数据增强**（`augment=True` 时）：
- 亮度抖动：Y 通道 × uniform(0.85, 1.15)
- 对比度抖动：Y 通道 histogram stretch ±10%

**DataLoader 配置**：
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=4)
h1_loader    = DataLoader(h1_dataset,    batch_size=16, shuffle=False, num_workers=4)
```

**验证**：
- 单个 batch 加载耗时 < 100ms
- tensor shape 正确
- augment=True 时输出图像亮度有随机变化

**依赖**：T4

---

### T7 — 损失函数

**目标**：新建损失函数模块，实现 GaussianNLL + BCE + lead mask。

**文件**：`tools/dashcam/train/losses.py`（新建）

**设计依据**：

本损失函数的设计由两个因素决定：模型输出格式和标注数据特性。

> **注意**：openpilot 原始训练代码未公开，我们无法确认 comma.ai 使用的损失函数。以下设计是基于模型输出格式的合理推断，不是对原始训练的复现。

1. **为什么用 GaussianNLL 而非 MSE？**
   - 从推理代码 `parse_model_outputs.py` 的 `parse_mdn()` 可知，模型连续值输出采用 `[mean, log_sigma]` 格式，解码时对 log_sigma 做 `exp()` 得到标准差
   - GaussianNLL = `0.5 * [log_sigma + (target - mean)² / exp(2·log_sigma)]` 是该概率输出格式对应的标准似然函数
   - 若用 MSE 则忽略了 sigma 分支，模型一半输出维度无训练信号，sigma 分支会退化为任意值

2. **为什么 log_sigma 需要 clamp [-3, 3]？**
   - 这是 MDN 训练的标准做法，防止两个方向的数值病态：
     - **下界 -3**（sigma≈0.05m）：若 log_sigma → -∞，sigma → 0，NLL 中的 `(target-mean)²/exp(2·log_sigma)` 项指数增长，导致梯度爆炸。0.05m 已小于车道线标注本身的噪声水平，足够表达高置信
     - **上界 3**（sigma≈20m）：若 log_sigma → +∞，模型可通过预测极大 sigma "作弊"来降低 loss 而不改善 mean 精度。20m 对驾驶场景中任何输出头都是足够宽松的不确定性上限
   - 区间 [-3, 3] 覆盖的 sigma 范围 [0.05m, 20m] 对所有输出头（车道线、路沿、前车、pose）都是合理的工作区间

3. **为什么 lane_lines/road_edges 位置损失要用教师 prob 加权？**
   - 训练数据包含低置信度帧（清洗阈值仅 0.05），这些帧的位置标签不可靠
   - 用教师 prob 作为权重：prob=0.9 → 位置损失权重 0.9，prob=0.1 → 权重 0.1，自动降低噪声标签的梯度影响
   - 概率头 BCE 损失不加权，确保模型学习完整的置信度分布（包括"何时该输出低置信度"）
   - 该设计消除了对硬置信度阈值的依赖，是知识蒸馏的标准做法

4. **为什么 lead_prob 用软目标 BCE + 截断线性加权位置损失？**

   基于对标注数据的统计分析（`tools/dashcam/analysis/lead_prob_distribution.py`），lead_prob 呈**双峰分布**：52.6% < 0.05（确定无车），16.3% ≥ 0.9（确定有车），中间区 [0.2, 0.8) 占 10.3%。

   - **lead_prob 训练目标用软概率而非二值化**：
     - BCE 梯度天然具有自动降权效果：`∂BCE/∂logit = sigmoid(logit) - target`，当 target=0.5（教师不确定）时梯度上界仅 0.5，不确定样本自动贡献更小的梯度
     - 二值化在 0.5 边界制造跳变（0.49→0, 0.51→1），对相似输入产生矛盾监督信号
     - 与 lane_lines_prob 处理方式保持一致，均使用教师原始软概率作为知识蒸馏目标
     - 保留信息的不可逆性：数据中保留原始概率，训练时可随时调整策略

   - **lead 位置损失用截断线性加权（而非与 lane_lines 相同的线性加权）**：
     - 与 lane_lines 不同，lead 在低 prob 时位置数据是纯噪声（无几何约束），不能像 lane_lines 那样用 prob 线性加权
     - 截断线性 `weight = clamp((prob - 0.3) / 0.7, 0, 1)` 在 prob < 0.3 时完全归零（排除 74.8% 的无车帧），prob > 0.3 时平滑增长
     - 相比硬阈值 mask（0.49→0, 0.51→1 的跳变），截断线性在边界处连续，prob=0.5 和 prob=0.9 的位置数据获得不同的信任度
     - lead 位置损失仍有 0.5 的全局降权（缓解 MDN 数值不稳定风险）

5. **各分量权重的优先级逻辑**（见 training_evaluation_methodology.md §4.3.3）：
   - 权重 1.0：lane_lines / lane_lines_prob / road_edges / lead_prob — LDW/FCW 核心输出，必须精确
   - 权重 0.5：lead 位置 — FCW 核心但 MDN 数值不稳定，降权保护训练稳定性
   - 权重 0.2：pose / road_transform / wide_from_device_euler — 辅助任务，提供有用梯度但不应主导训练方向

**核心类**：

```python
class MultiHeightLoss(nn.Module):
    """
    将模型 977 维扁平输出按 ONNX_OUTPUT_SLICES 切片，
    与标注 targets 逐头计算损失。
    """
    def __init__(self, loss_weights: dict):
        # loss_weights 见 training_evaluation_methodology.md §4.3.3

    def forward(self, pred_flat: Tensor, targets: dict) -> tuple[Tensor, dict]:
        """
        返回:
          total_loss: scalar
          loss_dict: {head_name: scalar} 各分量损失值（用于日志）
        """
```

**各分量损失**：

| 输出头 | 损失类型 | 特殊处理 |
|--------|---------|---------|
| lane_lines | GaussianNLL | log_sigma clamp [-3, 3]；**逐条置信度加权**（见下方说明） |
| road_edges | GaussianNLL | log_sigma clamp [-3, 3]；**逐条置信度加权**（见下方说明） |
| lead | GaussianNLL | **截断线性加权**：`weight = clamp((prob - 0.3) / 0.7, 0, 1)`，prob < 0.3 时位置损失为 0 |
| pose | GaussianNLL | log_sigma clamp [-3, 3] |
| road_transform | GaussianNLL | log_sigma clamp [-3, 3] |
| wide_from_device_euler | GaussianNLL | log_sigma clamp [-3, 3] |
| lane_lines_prob | BCE | 8 维 logit vs 软目标概率（**不加权，所有帧均参与**） |
| lead_prob | BCE | 3 维 logit vs **软目标概率**（与 lane_lines_prob 一致，不加权） |

**GaussianNLL 实现**：
```python
def gaussian_nll(pred_mean, pred_log_sigma, target):
    log_sigma = pred_log_sigma.clamp(-3.0, 3.0)
    return 0.5 * (log_sigma + (target - pred_mean)**2 / torch.exp(2 * log_sigma))
```

**置信度加权位置损失**（lane_lines / road_edges）：

由于训练数据包含低置信度帧，位置标签的可靠性与教师 prob 正相关。
使用教师 prob 作为逐条加权系数，自动降低不可靠位置标签的梯度贡献：

```python
# lane_lines: 4 条车道线，每条独立加权
for k in range(4):
    weight = targets['lane_lines_prob'][k]          # 教师置信度，range [0, 1]
    loss_k = gaussian_nll(pred_ll[k], pred_ll_sigma[k], target_ll[k])  # (33, 2)
    lane_pos_loss += weight * loss_k.mean()

# road_edges: 2 条路沿，同理
for k in range(2):
    weight = targets['road_edges_prob'][k]          # 固定 1.0（标注无逐条概率）
    loss_k = gaussian_nll(pred_re[k], pred_re_sigma[k], target_re[k])
    edge_pos_loss += weight * loss_k.mean()
```

效果：prob=0.9 → 位置损失权重 0.9；prob=0.1 → 权重 0.1（自动降权不可靠标注）。
概率头 BCE 损失**不加权**，始终学习完整的置信度分布。

**截断线性加权位置损失**（lead）：

lead 在低 prob 时位置数据缺乏几何约束（不像车道线有道路结构），因此不能用简单的线性加权，
需要更激进地排除低置信度帧。截断线性在 prob < 0.3 时完全归零，> 0.3 时平滑增长至 1.0：

```python
# lead: 3 个时间偏移，每个独立加权
for i in range(3):
    # 截断线性: prob<0.3 → 0, prob=0.3 → 0, prob=0.65 → 0.5, prob=1.0 → 1.0
    weight = torch.clamp((targets['lead_prob'][i] - 0.3) / 0.7, min=0.0, max=1.0)
    if weight > 0:
        loss_i = gaussian_nll(pred_lead[i], pred_lead_sigma[i], target_lead[i])  # (6, 4)
        lead_pos_loss += weight * loss_i.mean()
```

效果：74.8% 的 prob<0.3 帧完全不贡献位置梯度；prob=0.5 → 权重 0.29；prob=0.9 → 权重 0.86。

**验证**：
- 随机输入下 total_loss 为有限正数（无 NaN/Inf）
- lead_prob < 0.3 时 lead 位置损失 = 0
- lane_lines_prob=0.0 时，对应 lane 位置损失贡献 = 0
- 各分量损失量级合理（不超过 100）

**依赖**：无

---

### T8 — 训练主脚本

**目标**：新建训练主脚本，集成模型加载、数据加载、损失计算、渐进解冻、early stopping。

**文件**：`tools/dashcam/train/train_height.py`（新建）

**命令行接口**：
```bash
python tools/dashcam/train/train_height.py \
    --pretrained checkpoints/inadas_original.pt \
    --cache-dir data/caches/ \
    --output-dir checkpoints/implicit_v1/ \
    --epochs 100 \
    --batch-size 16 \
    --early-stop-patience 20 \
    --h1-degradation-threshold 0.05 \
    --augment
```

**核心功能**：

1. **数据加载**：
   - `train/` → 训练集（shuffle, augment）
   - `val/` → 验证集
   - `h1_holdout/` → H1 退化验证集

2. **渐进式解冻**（§4.1）：
   - Phase 切换由 epoch 边界 + 收敛检测触发
   - 切换时调用 `set_phase(model, phase)` + 更新 optimizer param_groups

3. **训练循环**（每 epoch）：
   ```
   a. 遍历 train_loader，前向 → 损失 → 反向 → 梯度裁剪 → optimizer.step
   b. 遍历 val_loader，计算 val_loss_total
   c. 遍历 h1_loader，计算 val_loss_H1
   d. scheduler.step()
   e. Early stopping 双指标判定
   f. 保存 best checkpoint（按 val_loss_total）
   g. 日志输出：epoch, train_loss, val_loss, val_loss_H1, lr, phase, 各分量损失
   ```

4. **Checkpoint 保存**：
   ```python
   {
     'epoch': int,
     'phase': int,
     'model_state_dict': ...,
     'optimizer_state_dict': ...,
     'scheduler_state_dict': ...,
     'val_loss': float,
     'val_loss_h1': float,
     'val_loss_h1_baseline': float,  # epoch 0 的 H1 loss
   }
   ```

5. **断点续训**：`--resume checkpoints/implicit_v1/last.pt`

**验证**：
- 1 epoch 训练不报错
- val_loss 有下降趋势
- val_loss_H1 被正确计算和记录
- checkpoint 文件可正常加载恢复

**依赖**：T5, T6, T7

---

## Phase C: 评价基础设施

### T9 — 离线评价脚本

**目标**：新建评价脚本，在测试集上计算 training_evaluation_methodology.md §5 定义的全部指标。

**文件**：`tools/dashcam/train/evaluate_height.py`（新建）

**命令行接口**：
```bash
python tools/dashcam/train/evaluate_height.py \
    --checkpoint checkpoints/implicit_v1/best.pt \
    --cache-dir data/caches/test/ \
    --h1-cache-dir data/caches/h1_holdout/ \
    --output checkpoints/implicit_v1/eval_report.txt
```

**输出内容**：

1. **分高度指标表**（§5.2）：
   ```
   Height  | y_lat MAE | y_lat [10,30] | y_lat [30,60] | z MAE | lane AUC | lead x MAE | lead AUC | RT h MAE
   ```

2. **LDW 功能评价**（§5.4）：
   - 横向偏移检测误差中位数、<0.1m 占比、<0.2m 占比

3. **FCW 功能评价**（§5.5）：
   - 前车检测率、纵向距离 MAE（按距离段）

4. **H1 退化检测**（§5.6）：
   - 与预训练基线对比的退化率

5. **达标判定**（§5.7.1）：
   - 各高度是否满足达标标准，总体判定 PASS/FAIL

**验证**：
- 预训练模型在 H1 测试集上的指标作为 sanity check
- H1 退化指标可正常计算
- 输出格式与 §5.2.5 一致

**依赖**：T5, T6（开发依赖）；运行时依赖 T4 生成的缓存数据

---

## Phase D: 训练执行与迭代

### T10 — 快速验证实验

**目标**：用少量数据（H1 + H6）验证全流水线可行性。

**操作**：
1. 从 `data/multi_height-0312/` 中选取 2–3 个 session（名义中心 pitch=5°, yaw=0°）
2. 对选中 sessions 执行 T2（清洗）→ T3（划分）→ T4（缓存）流水线，仅处理 H1 和 H6，生成小数据集到 `data/caches_mini/`
3. 运行 `train_height.py --cache-dir data/caches_mini/`，Phase 1 only，10 epochs
4. 运行 `evaluate_height.py --cache-dir data/caches_mini/test/ --h1-cache-dir data/caches_mini/h1_holdout/`，检查 H1/H6 指标

**成功标准**：
- 训练 loss 收敛（无爆炸、无 NaN）
- H1 退化 < 20%
- H6 y_lat MAE < 1.0m

**依赖**：T2–T9 全部

---

### T11 — 正式训练实验

**目标**：使用全量数据执行完整的渐进式微调。

**操作**：
1. 对全部 163 sessions 执行 T2→T3→T4 完整流水线，生成 `data/caches/` 全量缓存
2. 运行 `train_height.py --cache-dir data/caches/`，EXP-B 配置：Phase 1→3，100 epochs
3. 运行 `evaluate_height.py --cache-dir data/caches/test/ --h1-cache-dir data/caches/h1_holdout/` 生成完整评价报告

**依赖**：T10 成功

---

### T12 — 评价与决策

**目标**：根据 §5.7.1 达标标准判定是否需要进入显式方案。

**操作**：
1. 分析 T11 的评价报告
2. 对照 §5.7.1 达标表格逐高度检查
3. 做出决策：部署隐式模型 / 进入第二阶段

**依赖**：T11

---

## Phase E: 部署

### T13 — 导出与编译

**目标**：将训练好的 best.pt 导出为 ONNX 和 tinygrad pkl，用于在线推理。

**操作**：
```bash
# ONNX 导出
python tools/dashcam/train/export_onnx.py \
    --checkpoint checkpoints/implicit_v1/best.pt \
    --output checkpoints/implicit_v1/driving_vision.onnx \
    --dual-camera

# tinygrad 编译
DEV=CUDA python tools/dashcam/train/compile_tinygrad.py \
    checkpoints/implicit_v1/driving_vision.onnx
```

**验证**：
- ONNX 输出 shape = (1, 977)
- tinygrad pkl 推理结果与 PyTorch 一致（MAE < 0.01）
- `custom_modeld.py` 可正常加载并运行

**依赖**：T11

---

## 任务依赖图

```
T1 (批量标注) ✅ 已完成
 └─→ T2 (数据清洗)
      ├─→ T3.1 (训练集划分)
      └─→ T3.2 (H1 退化验证集)
           └─→ T4 (预处理缓存)
                └─→ T6 (Dataset)
                     └─→ T8 (训练脚本) ←── T5 (模型加载) + T7 (损失函数)
                          └─→ T10 (快速验证) ←── T9 (评价脚本)
                               └─→ T11 (正式训练)
                                    └─→ T12 (评价决策)
                                         └─→ T13 (导出部署)
```

**可并行的任务**：
- T5, T7 不依赖数据，可与 T2–T4 **并行开发**
- T9 不依赖数据，可与 T3–T4 **并行开发**（但运行时需要缓存数据）
- T3.1 和 T3.2 可**并行执行**（都仅依赖 T2 输出）

---

## 开发顺序建议

> T1（批量标注）已完成，从 T2 开始。

```
第 1 轮（并行）：
  ├── T2  数据清洗脚本
  ├── T5  模型加载与层级分析（不依赖数据）
  └── T7  损失函数（不依赖数据）

第 2 轮（T2 完成后）：
  ├── T3  数据划分（T3.1 + T3.2）
  └── T9  评价脚本（可用 T5 的模型 + 后续数据验证）

第 3 轮：
  ├── T4  预处理缓存
  └── T6  Dataset（等 T4 完成后验证）

第 4 轮：
  └── T8  训练主脚本（集成 T5, T6, T7）

第 5 轮：
  └── T10 快速验证实验

第 6 轮：
  └── T11 正式训练

第 7 轮：
  ├── T12 评价决策
  └── T13 导出部署
```
