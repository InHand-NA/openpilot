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

## 3. 设计决策

### 决策1：是否修改 warp 矩阵？

**结论：不修改 warp 矩阵，添加高度作为显式模型输入。**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 修改 warp 为完整 IPM（逆透视映射） | 地面特征对齐 | 3D 物体（车辆）严重失真；需要精确地面平面估计；推理管线大改 |
| 高度显式输入（HeightConditionedHead） | 轻量；模型可学习高度相关映射；与现有管线兼容 | 依赖高度输入的准确性 |

**选择方案2**：高度嵌入注入 backbone 输出层，模型自适应高度差异。

### 决策2：高度输入的注入方式？

**结论：高度嵌入注入 backbone 输出（977维特征）后，经过高度条件化 MLP 输出最终预测。**

```
PretrainedVisionModel(backbone, freeze)
    → flat: (B, 977)
    → [flat ‖ h_emb(8维)]
    → HeightConditionedHead(MLP)
    → 输出: (B, 977) 与原输出格式兼容
```

backbone 冻结保留 23M 预训练知识，只训练轻量级高度调制层（参数量 < 1%）。

### 决策3：标注策略？

**结论：分层策略——低高度用 modeld 标注，高高度用 Carla GT。**

| 高度范围 | 标注方案 | 原因 |
|---------|---------|------|
| 1.0–1.5m（接近标准高度） | modeld 在线标注（`--record-modeld`） | 模型精度高，标注可靠 |
| 1.5–3.0m（远离标准高度） | Carla GT + modeld 双标注对比 | 高高度下 modeld 精度下降，优先用地图 API 精确 GT |

---

## 4. 数据采集方案

### 4.1 多高度 Carla 配置

采集 **6 个高度档位**，覆盖主要车型：

| 档位 | 高度 H (m) | 车型 | 俯仰角策略 |
|------|-----------|------|-----------|
| H1   | 1.0       | 小型轿车 | pitch = arctan(H/15) ≈ 3.8° 向下 |
| H2   | 1.3       | 标准轿车/comma 3X | pitch ≈ 4.9° 向下（标准参考） |
| H3   | 1.5       | SUV / 越野 | pitch ≈ 5.7° 向下 |
| H4   | 2.0       | 中型货车 / 厢式面包车 | pitch ≈ 7.6° 向下 |
| H5   | 2.5       | 大型货车 / 中型卡车 | pitch ≈ 9.5° 向下 |
| H6   | 3.0       | 重卡 / 长途卡车 | pitch ≈ 11.3° 向下 |

俯仰角计算（保持 15m 前方为注视中心）：

```python
pitch_down = math.degrees(math.atan(height / 15.0))  # 向下看的俯仰角
```

**采集规模目标**：每高度档位 ≥ 5000 帧，覆盖 ≥ 3 个地图（Town04/Town05/Town07），总计 ≥ 30000 帧。

### 4.2 Carla 场景配置

每个高度档位的采集配置：

```python
HEIGHT_CONFIGS = {
    'H1': {'height': 1.0, 'pitch': -3.8, 'vehicle': 'vehicle.tesla.model3'},
    'H2': {'height': 1.3, 'pitch': -4.9, 'vehicle': 'vehicle.toyota.prius'},
    'H3': {'height': 1.5, 'pitch': -5.7, 'vehicle': 'vehicle.ford.mustang'},
    'H4': {'height': 2.0, 'pitch': -7.6, 'vehicle': 'vehicle.mercedes.sprinter'},
    'H5': {'height': 2.5, 'pitch': -9.5, 'vehicle': 'vehicle.carlamotors.firetruck'},
    'H6': {'height': 3.0, 'pitch': -11.3, 'vehicle': 'vehicle.carlamotors.european_hgv'},
}
```

场景条件：
- **天气**：晴天 / 多云 / 小雨（各 1/3）
- **时间**：日间（6:00–18:00）
- **道路类型**：双车道乡村路 + 高速公路 + 城区道路
- **交通密度**：中密度（50辆 NPC 车辆）

### 4.3 采集脚本修改

在 `tools/dashcam/carla_world.py` 的相机挂载点添加高度参数支持：

```python
# 在 carla_world.py 中增加 camera_height 参数
class CameraConfig:
    height: float = 1.22    # 相机离地面高度（米）
    pitch: float = -4.9     # 相机俯仰角（度）
    forward_offset: float = 2.0  # 前向偏移（米）

# 批量多高度采集入口
def collect_multi_height(heights, frames_per_height=5000, output_base='data/multi_height'):
    for h_config in heights:
        output_dir = f"{output_base}/H{h_config['height']:.1f}"
        run_collection(camera_height=h_config['height'],
                       camera_pitch=h_config['pitch'],
                       output_dir=output_dir,
                       n_frames=frames_per_height)
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

### 5.2 Carla GT 标注方法（H > 1.5m）

#### 车道线 GT

从 Carla 地图 API 提取 waypoints，投影到设备帧：

```python
def get_lane_lines_gt(world, vehicle, camera_height, n_points=33):
    """从 Carla waypoint API 提取车道线，投影到设备帧坐标系。

    坐标变换链：
        Carla 世界帧 → 车辆帧 → 设备帧（device frame）
    """
    map_api = world.get_map()
    vehicle_transform = vehicle.get_transform()

    lanes = []
    for lane_offset in [-1.5, -0.5, 0.5, 1.5]:  # 左右两条车道线各两侧
        waypoints = []
        wp = map_api.get_waypoint(vehicle_transform.location)
        for _ in range(n_points):
            # 获取 X_IDXS 对应距离的 waypoint
            wp_pos = wp.transform.location
            # 转换到车辆坐标系
            local_x, local_y, local_z = world_to_vehicle(wp_pos, vehicle_transform)
            # 添加车道偏移
            lane_y = local_y + lane_offset
            # 转换到设备帧（相机高度修正 z）
            device_z = local_z - camera_height
            waypoints.append([local_x, lane_y, device_z])
            wp = wp.next(get_step(local_x))[0]
        lanes.append(waypoints)
    return np.array(lanes)  # (4, 33, 3) [x, y_lat, z_height]
```

#### 前车 GT

从 traffic manager 获取 NPC 车辆的位置和速度：

```python
def get_lead_vehicle_gt(world, ego_vehicle):
    """获取前方车辆 GT，转换到设备帧坐标系。

    返回: lead (3, 6, 4) = [prob, x_fwd, y_lat, v_rel, a_rel, ...]
         lead_prob (3,) = [has_lead_0s, has_lead_2s, has_lead_4s]
    """
    ego_transform = ego_vehicle.get_transform()
    ego_velocity = ego_vehicle.get_velocity()

    npc_vehicles = world.get_actors().filter('vehicle.*')
    leads = []
    for npc in npc_vehicles:
        npc_transform = npc.get_transform()
        # 计算相对位置（设备帧）
        dx, dy, dz = world_to_device(npc_transform.location,
                                      ego_transform, camera_height)
        if dx > 0 and dx < 100 and abs(dy) < 3.0:  # 前方100m内，横向3m内
            npc_vel = npc.get_velocity()
            v_rel = (npc_vel.x - ego_velocity.x) * math.cos(ego_yaw)
            leads.append({'x': dx, 'y': dy, 'v_rel': v_rel, 'a_rel': 0.0})

    return format_lead_output(leads)  # 格式化为模型输出格式
```

#### 坐标变换链

```
Carla 世界帧 (右手, Z向上)
    → 车辆帧 (右手, X向前, Y向左, Z向上)
    → openpilot 设备帧 (右手, X向前, Y向左, Z向上)
    → 模型坐标 (lat=Y, height=Z)
```

关键变换参数：
- Carla 使用左手坐标系（Y 向右），需要翻转 Y 轴
- 高度分量需要减去相机安装高度：`z_device = z_carla - camera_height`

### 5.3 标注质量验证

对于 1.3m–1.5m 重叠区间，对比 modeld 标注与 Carla GT：

```python
# 验证指标
def validate_annotation_quality(modeld_labels, gt_labels, distance_bins=[30, 80, 192]):
    for head in ['lane_lines', 'lead']:
        for i, x_max in enumerate(distance_bins):
            x_min = distance_bins[i-1] if i > 0 else 0
            mask = (X_IDXS >= x_min) & (X_IDXS < x_max)
            mae = np.abs(modeld_labels[head][:, mask] - gt_labels[head][:, mask]).mean()
            print(f"{head} [{x_min}m–{x_max}m] MAE: {mae:.3f}m")
```

标注置信度过滤：
- 仅使用 `lane_lines_prob > 0.5` 的帧（modeld 置信度过滤）
- 丢弃 `v_ego < 5 m/s` 的低速帧（标注噪声较大）

---

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
| H1 (1.0m) | 采集 500 帧 | 测试集（接近标准，验证不退化） |
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
