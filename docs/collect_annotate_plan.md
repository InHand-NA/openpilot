# 多高度数据采集与标注软件工具开发规划

> 本文件是 [height_extension_design.md](height_extension_design.md) 第 4、5 节的软件实现规划。
> 目标：开发 Carla 多高度同步采集 + 离线标注流水线，为多高度模型微调提供训练数据。

---

## 目录

- [多高度数据采集与标注软件工具开发规划](#多高度数据采集与标注软件工具开发规划)
  - [目录](#目录)
  - [1. 总体数据流程](#1-总体数据流程)
  - [2. 目录结构设计](#2-目录结构设计)
  - [3. 软件工具规划（采集与标注）](#3-软件工具规划采集与标注)
    - [3.1 `carla_multi_height_world.py`（新建）](#31-carla_multi_height_worldpy新建)
    - [3.2 `collect_multi_height.py`（新建）](#32-collect_multi_heightpy新建)
    - [3.3 `annotate_multi_height.py`（新建）](#33-annotate_multi_heightpy新建)
    - [3.4 `preprocess_cache.py`（修改）](#34-preprocess_cachepy修改)
    - [3.5 `dataset.py`（修改）](#35-datasetpy修改)
  - [4. 可视化工具规划](#4-可视化工具规划)
    - [4.0 可视化基础设施与复用策略](#40-可视化基础设施与复用策略)
    - [4.1 `viz/browse_raw_session.py`（新建）](#41-vizbrowse_raw_sessionpy新建)
    - [4.2 `viz/inspect_annotated.py`（新建）](#42-vizinspect_annotatedpy新建)
    - [4.3 `viz/compare_heights.py`（新建）](#43-vizcompare_heightspy新建)
    - [4.4 `viz/label_stats.py`（新建）](#44-vizlabel_statspy新建)
  - [5. 快速验证阶段运行步骤](#5-快速验证阶段运行步骤)
  - [6. 实施顺序与依赖关系](#6-实施顺序与依赖关系)
  - [7. 关键设计决策记录](#7-关键设计决策记录)

---

## 1. 总体数据流程

```
┌──────────────────────────────────────────────────────────────────────┐
│  Step A：Carla 多高度同步采集                                          │
│  collect_multi_height.py + carla_multi_height_world.py               │
│                                                                      │
│  Carla 仿真（单次运行）                                                │
│    ├── H1 窄焦+宽焦相机 ──→ H1/{帧号}.npz  (road_rgb, wide_rgb, meta) │
│    ├── H2 窄焦+宽焦相机 ──→ H2/{帧号}.npz                             │
│    ├── ...                                                           │
│    └── H6 窄焦+宽焦相机 ──→ H6/{帧号}.npz                             │
│                                                                      │
│  clip_info.json：记录本 session 的 pitch, yaw, height, map, weather   │
└──────────────────────────────────────────┬───────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step B：离线标注（annotate_multi_height.py）                          │
│                                                                      │
│  H1/{帧号}.npz (road_rgb, wide_rgb)                                  │
│    → 从 clip_info.json 读取 pitch/yaw（方案A）                        │
│    → get_warp_matrix([0, pitch_rad, yaw_rad], NARROW_CAM_INTRINSICS) │
│    → PretrainedVisionModel 推理 → 977 维标注 flat_h1                  │
│    → 质量过滤：lane_lines_prob[1,2] > 0.5                             │
│    → transform_annotation(flat_h1, h1=1.22, h_k) × 5 次             │
│    → 写回 H1~H6/{帧号}.npz（追加标注字段）                             │
└──────────────────────────────────────────┬───────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step C：预处理缓存（preprocess_cache.py，已有，小幅修改）              │
│                                                                      │
│  H1~H6/{帧号}.npz（含 road_rgb, wide_rgb, 标注）                      │
│    → 从 clip_info.json 读 pitch/yaw（方案A，替代 rpyCalib）            │
│    → get_warp_matrix → warp → YUV420 (6,128,256) uint8               │
│    → 1FPS 抽样（每 20 帧保留 1 帧）                                    │
│    → H1~H6_cache/{帧号}.npz（road_yuv, wide_yuv, 标注张量）           │
└──────────────────────────────────────────┬───────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step D：模型训练（train.py，后续规划）                                 │
│  输入：H1~H6_cache/ 多目录                                            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构设计

```
data/multi_height/
  {session_tag}/               # 单次采集 session（一种 pitch×yaw×map×weather 组合）
    H1/                        # 标准高度（1.22m），窄+宽相机
      000001.npz               # 含 road_rgb, wide_rgb, camera_height, v_ego, world_pose
      000002.npz
      ...
    H2/                        # 1.3m
    H3/                        # 1.5m
    H4/                        # 2.0m
    H5/                        # 2.5m
    H6/                        # 3.0m
    clip_info.json             # session 元数据（见下方说明）

  {session_tag}_annotated/     # 标注完成后的副本（Step B 输出）
    H1/                        # road_rgb, wide_rgb, flat_h1（977维）, 质量标志
    H2/                        # road_rgb, wide_rgb, flat_h2（变换后）
    ...
    H6/

  {session_tag}_annotated/H1_cache/  # warp+YUV预处理缓存（Step C 输出）
  {session_tag}_annotated/H2_cache/
  ...
  {session_tag}_annotated/H6_cache/
```

**`clip_info.json` 关键字段：**
```json
{
  "session_id": "quick_H1H6_Town04_ClearNoon_p5.0_y0.0",
  "phase": "quick",
  "map": "Town04",
  "weather": "ClearNoon",
  "camera": {
    "pitch_deg": 5.0,
    "yaw_deg": 0.0,
    "forward_offset_m": 0.8
  },
  "heights": {
    "H1": 1.22,
    "H2": 1.3,
    "H6": 3.0
  },
  "simulation": {
    "fps": 20.0,
    "fixed_delta_seconds": 0.05,
    "num_npc": 40
  }
}
```

> `pitch_deg` / `yaw_deg` 是所有高度相机共用的安装角度，供标注器（Step B）和预处理器（Step C）读取（方案A：固定已知姿态）。

---

## 3. 软件工具规划（采集与标注）

### 3.1 `carla_multi_height_world.py`（新建）

**文件路径：** `tools/dashcam/carla_multi_height_world.py`

**职责：** 在单次 Carla 仿真中同时挂载多套（窄+宽）相机，按高度同步采集。

**核心类：**

```python
@dataclass
class CameraSlot:
    tag: str          # 'H1', 'H2', ..., 'H6'
    height: float     # 离地高度（米）

class MultiHeightCarlaWorld:
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 2000,
        town: str = 'Town04',
        spawn_point: int = 16,
        random_spawn: bool = False,
        camera_pitch_deg: float = 5.0,   # 所有高度共用同一 pitch
        camera_yaw_deg: float = 0.0,     # 所有高度共用同一 yaw
        camera_slots: list[CameraSlot],  # 要挂载的高度列表
        num_npc: int = 40,
        high_quality: bool = False,
        speed_range: tuple = (40.0, 100.0),
    )

    def get_frames(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """返回 {tag: (road_rgb, wide_rgb)}，未就绪返回 None"""

    def tick(self) -> None
    def get_vehicle_speed(self) -> float
    def get_vehicle_transform(self) -> carla.Transform
    def get_clip_metadata(self) -> dict
    def close(self) -> None
```

**帧同步设计（Carla 同步模式）：**

Carla 同步模式（`synchronous_mode = True`）在物理层面保证：**同一 `world.tick()` 触发的所有 sensor，其数据来自相同的物理状态，携带相同的 `image.frame` 帧序号**。12 个相机（6高度×2）只要 `sensor_tick` 相同，就一定产生于同一仿真 tick，天然帧对齐。

但 Carla 的 sensor 回调通过**独立流线程**（streaming thread）异步送达 Python，`world.tick()` 返回时不保证 12 个回调全部执行完毕。若以"H1 narrow 先到达"为同步基准（类似 `DashcamCarlaWorld._cam_callback_road` 第186行的做法），读取其他相机时其回调可能还未到达，产生帧号不一致的数据混用。

**正确方案：`image.frame` 帧号全匹配检查**

每个回调写入 `(frame_id, rgb)` 二元组，`get_frames()` 仅在所有相机都持有同一 `frame_id` 的数据时才返回，否则返回 `None`：

```python
# 回调（每个相机 slot × narrow/wide 各一个）
def _make_callback(self, tag: str, key: str):  # key = 'road' | 'wide'
    def callback(image: carla.Image):
        rgb = self._to_rgb(image)
        with self._lock:
            self._latest[tag][key] = (image.frame, rgb)
    return callback

# 采集循环调用
def get_frames(self) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """仅在所有 12 个相机（6高度×2）均已汇报当前帧时返回，否则返回 None。"""
    with self._lock:
        # 收集所有相机的最新帧号
        frame_ids = [
            self._latest[slot.tag][key][0]
            for slot in self._camera_slots
            for key in ('road', 'wide')
            if self._latest[slot.tag][key] is not None
        ]
        total = len(self._camera_slots) * 2
        if len(frame_ids) < total:
            return None  # 还有相机尚未收到任何数据
        if len(set(frame_ids)) != 1:
            return None  # 帧号不一致，等待滞后的回调
        # 所有相机帧号一致，安全返回
        return {
            slot.tag: (
                self._latest[slot.tag]['road'][1].copy(),
                self._latest[slot.tag]['wide'][1].copy(),
            )
            for slot in self._camera_slots
        }
```

**采集主循环：**
```python
self.tick()          # world.tick()，推进一步仿真
frames = None
while frames is None:
    frames = self.get_frames()   # 轮询直到 12 个回调全部到达
    # 无需 sleep：回调通常在 tick() 返回后数毫秒内全部到达
```

> **为何不会死等：** 在同步模式下，`world.tick()` 完成后 Carla 服务端已生成本 tick 所有 sensor 数据，客户端流线程必然在下一次 `tick()` 前收到全部回调。轮询不会漏帧，也不会跨 tick 混用数据。

> **与 `DashcamCarlaWorld` 的差异：** 现有 `DashcamCarlaWorld` 用 `_new_frame` 布尔标志（以 road camera 为基准），对 2 个相机窗口极短，实践中影响有限。对 12 个相机，滞后窗口扩大 6×，帧号不一致概率显著增加，必须使用 `image.frame` 全匹配方案。

**其他实现细节：**
- 每个高度对应 2 个相机 Actor（narrow FOV=40°，wide FOV=120°），`attach_to=ego_vehicle`
- 所有相机设置相同的 `sensor_tick = 1.0 / 20.0`（20 FPS），保证每 tick 同时触发
- `high_quality=False` 时关闭后处理效果（`enable_postprocess_effects=False`），减少渲染开销

**与现有代码的关系：**
- 不继承 `DashcamCarlaWorld`，单独实现（不含 VisionIPC/modeld 逻辑）
- 车辆生成、NPC、速度控制逻辑参考 `DashcamCarlaWorld`（`carla_world.py:44-176`）

---

### 3.2 `collect_multi_height.py`（新建）

**文件路径：** `tools/dashcam/collect_multi_height.py`

**职责：** 多高度批量数据采集主入口脚本。

**命令行接口：**

```
python tools/dashcam/collect_multi_height.py \
    --phase quick          # quick | full | custom
    --output-base data/multi_height
    --max-frames 20000     # 每 session 的 Carla 原始帧数（20FPS，默认20000）
    [--heights H1 H6]      # 仅用于 custom 模式
    [--map Town04]         # 仅用于 custom 模式
    [--weather ClearNoon]  # 仅用于 custom 模式
    [--pitch 5.0]          # 仅用于 custom 模式
    [--yaw 0.0]            # 仅用于 custom 模式
    [--num-npc 40]
    [--spawn-point 16]
    [--no-display]
    [--host 127.0.0.1] [--port 2000]
```

**Phase 配置：**

| phase | heights | scenes | pitch×yaw 组合 | max-frames/session |
|-------|---------|--------|----------------|-------------------|
| `quick` | H1, H6 | Town04/ClearNoon | (5°, 0°) 单一组合 | 20000（→1000 训练帧） |
| `full` | H1~H6 | §4.4 6种 | §4.3.3 34 种 | 8000（→400 训练帧/组合） |
| `custom` | 参数指定 | 参数指定 | 参数指定 | 参数指定 |

**输出：**

```
data/multi_height/
  quick_Town04_ClearNoon_p5.0_y0.0/
    H1/  H2/  ...  H6/        # 各含 000001.npz ~ 020000.npz
    clip_info.json
```

**每帧 NPZ 字段：**
```
road_rgb:     [1208, 1928, 3] uint8  ← 原始分辨率，未 warp
wide_rgb:     [1208, 1928, 3] uint8
camera_height: float32
v_ego:        float32  (m/s)
world_pose:   [6] float32  [x, y, z, roll, pitch, yaw]（Carla 世界坐标）
```

**注意：** 采集阶段**不运行 modeld**，不存储任何标注，只存原始图像。warp 和标注均在后续步骤完成。

---

### 3.3 `annotate_multi_height.py`（新建）

**文件路径：** `tools/dashcam/annotate_multi_height.py`

**职责：** 离线标注器。对采集目录中的 H1 图像批量推理，并将标注变换到所有高度，输出含标注的 NPZ。

**命令行接口：**

```
python tools/dashcam/annotate_multi_height.py \
    data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/ \
    --onnx checkpoints/inadas_original.onnx \
    --output data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0_annotated/ \
    [--batch-size 8]
    [--min-ll-prob 0.5]    # §5.4 Step3 质量过滤阈值
    [--device cuda]
    [--heights H2 H3 H6]  # 只生成指定高度（默认全部）
```

**处理流程：**

> **重要前提：** `PretrainedVisionModel.forward()` 返回 `dict[str, Tensor]`（见 `pretrained_model.py:122`），
> 不是 977 维 flat tensor。每个键对应 ONNX 的一个输出切片，格式均为原始 MDN flat 向量。

```python
# Step 1: 加载配置
clip_info_src = session_dir / 'clip_info.json'
clip_info = json.load(clip_info_src)
pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
yaw_rad   = math.radians(clip_info['camera']['yaw_deg'])

# clip_info.json 复制到输出目录，供下游 preprocess_cache.py 通过
# Path(npz).parent.parent / 'clip_info.json' 定位（output_dir/{H_k}/{帧}.npz → output_dir/）
shutil.copy(clip_info_src, output_dir / 'clip_info.json')

# 计算 H1 warp 矩阵（方案A：固定已知姿态）
rpyCalib_h1 = [0, pitch_rad, yaw_rad]   # [roll, pitch, yaw] 弧度
warp_h1_narrow = get_warp_matrix(rpyCalib_h1, NARROW_CAM_INTRINSICS)
warp_h1_wide   = get_warp_matrix(rpyCalib_h1, WIDE_CAM_INTRINSICS, bigmodel_frame=True)

# Step 2: 加载 PretrainedVisionModel
model = PretrainedVisionModel('checkpoints/inadas_original.onnx')
model.eval().to(device)

X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float32)  # (33,) 前向距离序列

# Step 3: 按帧处理
for frame_npz in sorted(H1_dir.glob('*.npz')):
    data = np.load(frame_npz)
    road_yuv_tensor = torch.from_numpy(rgb_to_modeld_input(data['road_rgb'], warp_h1_narrow)).unsqueeze(0)
    wide_yuv_tensor = torch.from_numpy(rgb_to_modeld_input(data['wide_rgb'], warp_h1_wide)).unsqueeze(0)

    with torch.no_grad():
        outputs_h1 = model(road_yuv_tensor, wide_yuv_tensor)  # dict[str, Tensor]

    # 解码为 canonical 标注格式（与 extract_targets() / modeld_label_extractor 兼容）
    canonical_h1 = decode_model_output(outputs_h1, X_IDXS)

    # Step 4: 质量过滤（§5.4 Step 3）
    # lane_lines_prob 原始 (8,)=(4,2) logits，真实概率取第 1 列 sigmoid
    # 检查内侧两条车道线 L0(idx=1) 和 R0(idx=2)
    ll_prob = canonical_h1['lane_lines_prob']   # (4,) 已 sigmoid，来自 decode_model_output
    if not (ll_prob[1] > min_ll_prob and ll_prob[2] > min_ll_prob):
        continue

    # Step 5: 为每个高度生成标注
    for h_tag, h_k in heights.items():
        canonical_hk = transform_annotation(canonical_h1, h1=1.22, h_k=h_k)
        src_npz = (session_dir / h_tag / frame_npz.name)
        out = dict(np.load(src_npz))
        out.update(canonical_hk)          # 写入标注字段（与旧格式兼容）
        out['label_source'] = b'pretrained_h1'
        np.savez_compressed(output_dir / h_tag / frame_npz.name, **out)
```

**`decode_model_output()` 辅助函数：**

将 `PretrainedVisionModel` 的 dict 输出解码为 canonical 格式，与 `extract_targets()` 完全兼容。
MDN 布局遵循 openpilot 约定（`custom_modeld.py:decode_outputs` 为参考实现）：
`[all_means | all_log_sigma]`（前一半为均值，后一半为 log 标准差）。

```python
def decode_model_output(outputs: dict[str, torch.Tensor], X_IDXS: np.ndarray) -> dict[str, np.ndarray]:
    """从 PretrainedVisionModel dict 输出解码为 canonical 标注格式。

    MDN 格式（非交错）：[all_means | all_log_sigma]
    lane_lines (528,)   = 264 means + 264 log_sigma，means reshape (4,33,2) = [y, z]
    road_edges (264,)   = 132 means + 132 log_sigma，means reshape (2,33,2) = [y, z]
    lead (144,)         = 72 means + 72 log_sigma，  means reshape (3,6,4) = [x,y,v,a]
    pose (12,)          = 6 means + 6 log_sigma，    means = [trans(3), rot(3)]
    road_transform (12,)= 6 means + 6 log_sigma，    means[:3] = [tx, ty, tz]
    lane_lines_prob (8,)= (4,2) raw logits，真实概率 = sigmoid([:,1])
    """
    def _np(t): return t[0].cpu().numpy()  # (1, N) → (N,)

    # lane_lines: (528,) → means (4,33,2) [y,z] → 拼 x 列 → (4,33,3) [x,y,z]
    ll_raw = _np(outputs['lane_lines'])     # (528,)
    ll_means = ll_raw[:264].reshape(4, 33, 2)   # [y, z] per point
    x_col = np.broadcast_to(X_IDXS[None, :, None], (4, 33, 1))
    lane_lines = np.concatenate([x_col, ll_means], axis=-1).astype(np.float32)  # (4,33,3)

    # lane_lines_prob: (8,) → (4,2) logits → sigmoid([:,1]) → (4,)
    ll_prob_raw = _np(outputs['lane_lines_prob']).reshape(4, 2)
    lane_lines_prob = (1.0 / (1.0 + np.exp(-np.clip(ll_prob_raw[:, 1], -20, 20)))).astype(np.float32)

    # road_edges: (264,) → means (2,33,2) [y,z] → 拼 x 列 → (2,33,3)
    re_raw = _np(outputs['road_edges'])     # (264,)
    re_means = re_raw[:132].reshape(2, 33, 2)
    x_col2 = np.broadcast_to(X_IDXS[None, :, None], (2, 33, 1))
    road_edges = np.concatenate([x_col2, re_means], axis=-1).astype(np.float32)  # (2,33,3)

    # road_edges_prob: modeld 不输出 per-edge 概率，全 1.0
    road_edges_prob = np.ones(2, dtype=np.float32)

    # lead: (144,) → means (3,6,4) [x,y,v,a]
    ld_raw = _np(outputs['lead'])           # (144,)
    lead = ld_raw[:72].reshape(3, 6, 4).astype(np.float32)

    # lead_prob: (3,) → sigmoid → (3,)
    lp_raw = _np(outputs['lead_prob'])
    lead_prob = (1.0 / (1.0 + np.exp(-np.clip(lp_raw, -20, 20)))).astype(np.float32)

    # pose: (12,) means [:6] = [trans(3), rot(3)]
    pose = _np(outputs['pose'])[:6].astype(np.float32)

    # road_transform: (12,) means [:6]，其中 means[:3] = [tx, ty, tz]
    road_transform = _np(outputs['road_transform'])[:6].astype(np.float32)

    # wide_from_device_euler: (6,) means [:3] = [roll, pitch, yaw]
    wfde = _np(outputs['wide_from_device_euler'])[:3].astype(np.float32)

    return {
        'lane_lines':             lane_lines,        # (4, 33, 3) float32
        'lane_lines_prob':        lane_lines_prob,   # (4,) float32
        'road_edges':             road_edges,         # (2, 33, 3) float32
        'road_edges_prob':        road_edges_prob,   # (2,) float32
        'lead':                   lead,              # (3, 6, 4) float32
        'lead_prob':              lead_prob,          # (3,) float32
        'pose':                   pose,              # (6,) float32
        'road_transform':         road_transform,    # (6,) float32
        'wide_from_device_euler': wfde,              # (3,) float32
    }
```

**`transform_annotation()` 实现（§5.4 Step 2）：**

输入/输出均为 `decode_model_output()` 返回的 canonical dict，格式与 `extract_targets()` 兼容。
z_height 是 lane_lines/road_edges 的第 2 列（index=2），road_transform 的第 2 个 mean（tz）。

```python
def transform_annotation(canonical: dict[str, np.ndarray], h1: float, h_k: float) -> dict[str, np.ndarray]:
    """将 canonical 标注的 z_height 分量从高度 h1 变换到 h_k。

    仅修正相机高度差引起的 z 偏移（近场近似，X > 10m 有效）。
    """
    delta_h = h_k - h1
    out = {k: v.copy() for k, v in canonical.items()}
    # lane_lines: (4, 33, 3) [x, y, z]，z 为列索引 2
    out['lane_lines'][:, :, 2] += delta_h
    # road_edges: (2, 33, 3) [x, y, z]，z 为列索引 2
    out['road_edges'][:, :, 2] += delta_h
    # road_transform: (6,) means = [tx, ty, tz, rx, ry, rz]，tz 为索引 2
    out['road_transform'][2] += delta_h
    return out
```

> **MDN 偏移推导（以 lane_lines 为例）：**
> - ONNX 完整切片：`[117:645]`（528 维 = 264 means + 264 log_sigma）
> - means 子集：ONNX`[117:381]`（264 维），reshape `(4, 33, 2)` = `[y_lat, z_height]`
> - road_edges 完整：ONNX`[653:917]`（264 维 = 132 means + 132 log_sigma）
> - road_transform means：ONNX`[105:111]`（前 6 维），`[2]` = tz
>
> 以上由 `pretrained_model.py:ONNX_OUTPUT_SLICES` 和 `custom_modeld.py:decode_outputs` 联合确认。

**输出标注格式（与 `extract_targets()` 完全兼容，可直接送入 `preprocess_cache.py`）：**

| 字段 | 形状 | 说明 |
|------|------|------|
| `lane_lines` | `(4, 33, 3)` float32 | `[x, y_lat, z_height]`，x = X_IDXS |
| `lane_lines_prob` | `(4,)` float32 | sigmoid 后的真实概率 |
| `road_edges` | `(2, 33, 3)` float32 | `[x, y_lat, z_height]` |
| `road_edges_prob` | `(2,)` float32 | 全 1.0（modeld 无 per-edge 概率） |
| `lead` | `(3, 6, 4)` float32 | means `[x, y, v, a]` |
| `lead_prob` | `(3,)` float32 | sigmoid 后概率 |
| `pose` | `(6,)` float32 | `[trans(3), rot(3)]` means |
| `road_transform` | `(6,)` float32 | `[tx, ty, tz, 0, 0, 0]` means |
| `wide_from_device_euler` | `(3,)` float32 | wide camera euler angles means |
| `label_source` | bytes | `b'pretrained_h1'` |
| `road_rgb`, `wide_rgb` | 原始 | 保留自 H_k 源 NPZ |
| `camera_height`, `v_ego`, `world_pose` | 原始 | 保留自 H_k 源 NPZ |

---

### 3.4 `preprocess_cache.py`（修改）

**文件路径：** `tools/dashcam/train/preprocess_cache.py`

**修改内容：**

1. **支持方案A（固定已知姿态）warp：**
   - 读取 `clip_info.json`（由 `annotate_multi_height.py` 复制到 annotated 目录，§3.3）
   - 若存在，优先用 `camera.pitch_deg / camera.yaw_deg` 计算 warp
   - 若不存在（旧数据），回退到现有的 `npz['rpyCalib']`

2. **标注字段兼容（无需新增提取函数）：**
   - `annotate_multi_height.py` 输出的 annotated NPZ 已存储 canonical 格式字段（lane_lines (4,33,3)、road_edges (2,33,3) 等），与现有 `extract_targets()` 完全兼容
   - **无需 `extract_targets_from_flat()`**，直接调用 `extract_targets(data)` 即可

3. **保留 `camera_height` 到缓存 NPZ：**
   - 缓存 NPZ 中添加 `camera_height: float32` 字段
   - 供 dataset.py 读取，为后续显式高度注入（HeightConditionedHead）预留接口

**修改后的 `_process_one()` 逻辑：**
```python
# 读取 warp 参数（方案A优先）
# npz_path 形如 session_annotated/H1/000001.npz
# → parent.parent = session_annotated/
# annotate_multi_height.py 会把 clip_info.json 复制到 session_annotated/，路径因此可达
clip_info_path = Path(npz_path).parent.parent / 'clip_info.json'
if clip_info_path.exists():
    info = json.load(clip_info_path)
    pitch_rad = math.radians(info['camera']['pitch_deg'])
    yaw_rad   = math.radians(info['camera']['yaw_deg'])
    rpyCalib = np.array([0.0, pitch_rad, yaw_rad])
else:
    rpyCalib = data['rpyCalib'].astype(np.float64)  # 回退（旧数据）

# 提取标注：annotated NPZ 已是 canonical 格式，直接调用现有函数
targets = extract_targets(data)
targets['camera_height'] = data.get('camera_height', np.float32(1.22))
```

---

### 3.5 `dataset.py`（修改）

**文件路径：** `tools/dashcam/train/dataset.py`

**修改内容：**

1. **`extract_targets()` 和 `CachedDualCameraDrivingDataset` 返回 `camera_height`：**
   - 从缓存 NPZ 加载 `camera_height` 字段（缺省 1.22m）
   - 包含在返回的 targets dict 中

**数据集变化影响评估：**
- `camera_height` 目前不进入 loss 计算，仅作为额外信息字段存储
- 不影响现有训练流程；为第二阶段显式高度注入（HeightConditionedHead）预留接口



---

## 4. 可视化工具规划

> **设计原则**：可视化工具覆盖流水线的三个关键检查点——原始采集数据、离线标注结果、多高度对比——帮助人工快速定位数据质量问题，无需运行训练。

### 4.0 可视化基础设施与复用策略

**现有可复用资产（`tools/dashcam/`）：**

| 文件 | 提供的能力 | 新工具中的复用方式 |
|------|-----------|-----------------|
| `visualizer.py` | `project_points_to_image()`、`_build_transform()`、车道线多边形渲染、BEV 面板绘制 | 直接 import，用于标注叠加投影 |
| `view_dual_data.py` | `_draw_lane_lines()`、`_draw_road_edges()`、`_draw_leads()`、`_draw_info_panel()`、`_draw_bev_panel()`、`_check_warnings()`、键盘导航框架 | 核心绘图函数直接 import；键盘导航模式复制 |

**新增目录：** `tools/dashcam/viz/`（含 `__init__.py`），避免与现有脚本混淆。

**标注数据兼容性约定：**

`annotate_multi_height.py` 输出的 NPZ 直接存储 canonical 格式字段（由 `decode_model_output()` + `transform_annotation()` 生成），**无 `flat_label` 中间格式**：
```
lane_lines:             [4, 33, 3] float32  — [x, y_lat, z_height]，x = X_IDXS
lane_lines_prob:        [4] float32         — sigmoid 后的真实概率
road_edges:             [2, 33, 3] float32
road_edges_prob:        [2] float32         — 全 1.0
lead:                   [3, 6, 4] float32   — MDN means [x,y,v,a]
lead_prob:              [3] float32
pose:                   [6] float32         — [trans(3), rot(3)] means
road_transform:         [6] float32         — [tx, ty, tz, 0, 0, 0] means
wide_from_device_euler: [3] float32
label_source:           bytes               — b'pretrained_h1'
```

此格式与 `extract_targets()`（`dataset.py`）完全兼容，`preprocess_cache.py` 无需特殊处理。
`view_dual_data.py` 的全部绘图函数也可直接复用于新工具（field 形状一致）。

---

### 4.1 `viz/browse_raw_session.py`（新建）

**检查点：Step A（Carla 采集）完成后**

**目的：** 浏览多高度原始采集数据，验证所有相机正常工作、各高度图像符合预期，发现卡帧/过曝/分辨率异常等问题。

**命令行：**
```bash
python tools/dashcam/viz/browse_raw_session.py \
    data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0/
    [--heights H1 H4 H6]   # 默认显示全部已有高度
    [--start 0]
    [--wide-road] #查看广角相机rgb
```

**显示布局（默认：网格模式）：**
```
┌──────────────┬──────────────┬──────────────┐
│  H1 road_rgb │  H2 road_rgb │  H3 road_rgb │
│   (1.22m)    │   (1.3m)     │   (1.5m)     │
├──────────────┼──────────────┼──────────────┤
│  H4 road_rgb │  H5 road_rgb │  H6 road_rgb │
│   (2.0m)     │   (2.5m)     │   (3.0m)     │
└──────────────┴──────────────┴──────────────┘
             底部状态栏
  Frame: 001234/020000  v_ego: 22.3 m/s (80.4 km/h)
  pitch: 5.0°  yaw: 0.0°  map: Town04  weather: ClearNoon
```

每个面板左上角覆盖标签（`H1 1.22m`），右上角显示帧号。底部状态栏来自 `clip_info.json` + 当前帧的 `v_ego`。

**键盘操作：**

| 键 | 功能 |
|----|------|
| `←` / `→` | 前/后一帧 |
| `PgUp` / `PgDn` | ±20 帧（等于 1 秒仿真） |
| `Home` / `End` | 首/末帧 |
| `f` | 跳转到指定帧号（输入对话框） |
| `w` | 切换到 wide_rgb（与 road_rgb 交替显示） |
| `i` | 切换信息面板（帧元数据） |
| `s` | 截图 PNG |
| `q` / `ESC` | 退出 |



---

### 4.2 `viz/inspect_annotated.py`（新建）

**检查点：Step B（离线标注）完成后**

**目的：** 在 **warp 后的模型输入图像**（512×256）上叠加标注，验证 H1 推理质量和 H_k 变换后标注的正确性。

> **为何用 warp 后图像而非原始图像？** 标注是在 calibrated frame 下输出的，投影到 warp 后图像才符合模型训练时的视角，便于判断标注与图像内容是否对齐。

**命令行：**
```bash
python tools/dashcam/viz/inspect_annotated.py \
    data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0_annotated/ \
    [--height H1]          # 默认 H1；可切换
    [--start 0]
    [--filter-low]         # 仅显示低置信度帧（用于核查过滤决策）
    [--min-ll-prob 0.5]    # 质量过滤阈值（与 annotator 保持一致）
```

**显示布局（单高度，全宽模式）：**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  左侧：warp 后 road 图像 (512×256, 2× 放大 = 1024×512)                  │
│    · 车道线多边形叠加（颜色=置信度）                                       │
│    · 路沿线                                                              │
│    · 前车三角形 + 距离/速度标注                                                  │
│    · 右上角：当前高度标签（H1 1.22m / H4 2.0m）                           │
│    · 左上角：信息面板（帧号/置信度/是否通过过滤）                           │
│                                                                         │
│  右侧：BEV 俯视图 (300×450)                                             │
│    · 车道线鸟瞰                                                          │
│    · 前车位置点                                                          │
└─────────────────────────────────────────────────────────────────────────┘
底部：lane_lines_prob 条形图  [0.92] [0.88] [0.71] [0.05]  PASS/FAIL 标志
```

**置信度颜色编码：**
- `prob ≥ 0.8`：绿色
- `0.5 ≤ prob < 0.8`：黄色
- `prob < 0.5`：红色（过滤候选）
- 被过滤帧（未通过 min_ll_prob）：在左上角显示红色 `[FILTERED]` 横幅

**键盘操作（继承 `view_dual_data.py` 惯例）：**

| 键 | 功能 |
|----|------|
| `←` / `→` | 前/后一帧 |
| `PgUp` / `PgDn` | ±10 帧 |
| `Home` / `End` | 首/末帧 |
| `h` | 循环切换高度（H1 → H2 → ... → H6 → H1） |
| `l` | 切换车道线叠加 |
| `e` | 切换路沿叠加 |
| `v` | 切换前车叠加 |
| `i` | 切换信息面板 |
| `b` | 切换 BEV 面板 |
| `r` | 在 warp 后图像 / 原始图像之间切换（对比 warp 效果） |
| `f` | 切换仅显示过滤帧/全部帧 |
| `s` | 截图 |
| `q` / `ESC` | 退出 |

---

### 4.3 `viz/compare_heights.py`（新建）

**检查点：Step B（离线标注）完成后，核心工具**

**目的：** 同一帧下并排展示 H1～H6 六个高度的 **warp 后图像 + 标注**，直观对比不同视角下的感知效果和 `transform_annotation` 的变换正确性。这是验证多高度标注一致性的关键可视化工具。

**命令行：**
```bash
python tools/dashcam/viz/compare_heights.py \
    data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0_annotated/ \
    [--heights H1 H4 H6]   # 可指定子集（默认全部已有高度）
    [--start 0]
    [--no-annotations]     # 只显示原始 warp 图像（不叠加标注）
```

**显示布局（6高度，2×3 网格）：**
```
┌──────────────────────────────────────────────────────────────────────┐
│ H1 (1.22m) warp+标注 │ H2 (1.3m) warp+标注  │ H3 (1.5m) warp+标注  │
│  lane_prob: 0.92/0.88 │  lane_prob: 0.91/0.87 │  lane_prob: 0.90/0.85│
├──────────────────────────────────────────────────────────────────────┤
│ H4 (2.0m) warp+标注  │ H5 (2.5m) warp+标注  │ H6 (3.0m) warp+标注  │
│  lane_prob: 0.90/0.87 │  lane_prob: 0.90/0.87 │  lane_prob: 0.90/0.87│
└──────────────────────────────────────────────────────────────────────┘
Frame: 001234/020000  v_ego: 22.3 m/s  pitch: 5.0°  yaw: 0.0°
```

每个面板：
- 背景：warp 后 road 图像（512×256 → 缩放适应面板）
- 标注叠加：车道线多边形（颜色=置信度） + 路沿 + 前车
- 面板标题：高度标签 + `lane_lines_prob` 数值
- 过滤状态：PASS（绿框）/ FAIL（红框）

**可视化目的说明（在代码注释和文档中体现）：**
1. **验证 warp 一致性**：所有高度的 pitch 相同（5°），warp 后图像的地平线应位置一致
2. **验证 z_height 变换**：H_k 的车道线 z 分量 = H1 值 + ΔH，可通过悬停/打印数值验证
3. **发现标注异常**：如某高度车道线漂移、前车距离突变等
4. **直觉化理解近场盲区**：H5/H6 的近场（X<10m）区域空白，与 §2.6 分析吻合

**键盘操作：**

| 键 | 功能 |
|----|------|
| `←` / `→` | 前/后一帧 |
| `PgUp` / `PgDn` | ±10 帧 |
| `Home` / `End` | 首/末帧 |
| `a` | 切换标注叠加（开/关） |
| `z` | 切换 z_height 数值显示（在每条车道线上标注 z 值范围） |
| `l` / `e` / `v` | 分别切换车道线/路沿/前车图层 |
| `Tab` | 在 2×3 网格 / 单高度全幅 之间切换（聚焦单个高度） |
| `1`-`6` | 单高度全幅模式下选择 H1-H6 |
| `s` | 截图（保存整个网格） |
| `q` / `ESC` | 退出 |

---

### 4.4 `viz/label_stats.py`（新建）

**检查点：Step B（离线标注）完成后，定量质量审计**

**目的：** 生成各高度数据质量的统计报告，量化标注置信度分布、过滤率、z_height 变换一致性等指标，为训练前的数据配置提供决策依据。

**命令行：**
```bash
python tools/dashcam/viz/label_stats.py \
    data/multi_height/quick_Town04_ClearNoon_p5.0_y0.0_annotated/ \
    [--output stats_report/]   # 输出图表目录（默认：输入目录下 stats/）
    [--min-ll-prob 0.5]        # 质量过滤阈值
    [--sample 1000]            # 随机采样帧数（加速，0=全部）
```

**生成内容（`stats/` 目录）：**

**1. `lane_prob_distribution.png`**：各高度 `lane_lines_prob` 分布
- 子图网格（H1~H6 各一个子图）
- 每个子图：左右内侧车道线（L0/R0）的概率直方图
- 纵线标注：0.5 阈值位置

**2. `filter_rates.png`**：各高度过滤率柱状图
- X 轴：高度档位 H1~H6
- Y 轴：通过 min_ll_prob 阈值的帧占比
- 参考线：目标值（如 80%），低于此值标红

**3. `z_height_transform.png`**：z_height 变换验证
- X 轴：X_IDXS 距离（0~192m）
- Y 轴：均值 z_height（标定帧）
- 每条曲线对应一个高度档位
- 验证：各档位曲线应近似平行，间距 ≈ ΔH（±0.1m 容差）
- 不平行 → transform_annotation 实现有误

**4. `lead_detection_rate.png`**：前车检测率折线图
- X 轴：高度档位
- Y 轴：至少检测到 1 个前车（lead_prob > 0.3）的帧占比
- 可用于判断 NPC 密度是否足够

**5. `stats_summary.txt`**：纯文本摘要
```
=== 多高度标注质量统计摘要 ===
Session: quick_Town04_ClearNoon_p5.0_y0.0_annotated
分析帧数: 20000 (采样: 1000)
------------------------------------------
         总帧数  PASS帧  过滤率  L0_prob(中位)  R0_prob(中位)  Lead率
H1 1.22m  20000   18642   93.2%       0.91           0.89       72.1%
H2 1.30m  20000   18501   92.5%       0.91           0.89       72.1%  ← z变换
H3 1.50m  20000   18120   90.6%       0.90           0.88       72.1%  ← z变换
H4 2.00m  20000   17803   89.0%       0.90           0.87       72.1%  ← z变换
H5 2.50m  20000   17390   86.9%       0.90           0.87       72.1%  ← z变换
H6 3.00m  20000   16820   84.1%       0.90           0.87       72.1%  ← z变换
------------------------------------------
z_height 变换一致性: ✅ PASS (均值误差 < 0.02m vs. ΔH)
建议 min_ll_prob 阈值: 0.50  (H6 过滤后剩余 16820 帧)
```

> **关键用途**：若 H6 过滤后帧数 < 8000（§4.5 目标），需增加采集量；若 z_height 变换一致性 FAIL，需检查 `transform_annotation()` 实现。

---

## 5. 快速验证阶段运行步骤

以下是快速验证（H1+H6，Town04，ClearNoon，pitch=5°，yaw=0°）的完整运行步骤，含各阶段人工检查命令：

```bash
SESSION="Town04_ClearNoon_p5.0_y0.0"
SESSION_DIR="data/multi_height/${SESSION}"
ANNOTATED_DIR="${SESSION_DIR}_annotated"

# ── Step A：采集 ──────────────────────────────────────────────────────
# 需 Carla 服务器运行，约 17 分钟仿真（20000帧 @20FPS）
python tools/dashcam/collect_multi_height.py \
    --phase quick \
    --output-base data/multi_height \
    --max-frames 20000 \
    --no-display

# 【人工检查 A】浏览原始数据，确认 H1/H6 图像正常
python tools/dashcam/viz/browse_raw_session.py ${SESSION_DIR}
# 检查要点：
#   · 两个高度图像均有内容（无全黑/全白帧）
#   · H6 图像明显比 H1 视角更高（地平线更低，近场地面更少）
#   · v_ego 有合理速度变化（非固定常数）

# ── Step B：离线标注 ──────────────────────────────────────────────────
# 需 GPU，约 5~10 分钟处理 20000 帧
python tools/dashcam/annotate_multi_height.py \
    ${SESSION_DIR} \
    --onnx checkpoints/inadas_original.onnx \
    --output ${ANNOTATED_DIR} \
    --device cuda

# 【人工检查 B1】查看标注质量，确认车道线叠加合理
python tools/dashcam/viz/inspect_annotated.py ${ANNOTATED_DIR} --height H1
# 按 h 切换到 H6 复查；按 f 只看被过滤帧，理解过滤原因

# 【人工检查 B2】多高度对比，验证 transform_annotation 正确性
python tools/dashcam/viz/compare_heights.py ${ANNOTATED_DIR}
# 检查要点：
#   · 6个面板地平线位置一致（pitch 相同，warp 一致）
#   · H6 近场 X<10m 区域确实空白（近场盲区）
#   · 所有高度车道线横向位置（y_lat）一致，仅 z 分量随高度递增
#   · 按 z 键打开 z_height 数值显示，核对 H4~H6 值 ≈ H1值 + ΔH

# 【人工检查 B3】生成统计报告，确认数据量和质量满足训练要求
python tools/dashcam/viz/label_stats.py ${ANNOTATED_DIR}
# 检查要点：
#   · H1/H6 过滤后剩余帧数 ≥ 800（快速验证目标）
#   · z_height 变换一致性 PASS
#   · lane_lines_prob 中位值 > 0.7

# ── Step C：预处理缓存 ────────────────────────────────────────────────
# 32核，约 1~2 分钟
for H in H1 H6; do
    python tools/dashcam/train/preprocess_cache.py \
        ${ANNOTATED_DIR}/${H}
done

# ── Step D：训练 ──────────────────────────────────────────────────────
python tools/dashcam/train/train.py \
    --cache-dirs \
        ${ANNOTATED_DIR}/H1_cache \
        ${ANNOTATED_DIR}/H6_cache \
    --output-dir checkpoints/implicit_quick_v1 \
    --epochs 50 --batch-size 16 --early-stop 15
```

---

## 6. 实施顺序与依赖关系

```
carla_multi_height_world.py  ──→  collect_multi_height.py
                                         │
                              ┌──────────┤
                              ▼          ▼
             browse_raw_session.py   annotate_multi_height.py  ←── PretrainedVisionModel（已有）
             （Step A 检查）              │
                                ┌────────┤
                                ▼        ▼
              inspect_annotated.py   compare_heights.py    label_stats.py
              （Step B 检查）          （Step B 对比）      （Step B 统计）
                                         │
                                         ▼
                              preprocess_cache.py（修改）
                                         │
                                         ▼
                               dataset.py（修改）
                                         │
                                         ▼
                                train.py（暂无修改）
```

**建议开发顺序：**

| 优先级 | 工具 | 估计工作量 | 前置条件 |
|--------|------|-----------|---------|
| P1 | `carla_multi_height_world.py` | 2h | 无 |
| P1 | `collect_multi_height.py` | 2h | carla_multi_height_world.py |
| P1 | `viz/browse_raw_session.py` | 1.5h | collect_multi_height.py（有数据） |
| P2 | `annotate_multi_height.py` | 3h | PretrainedVisionModel（已有） |
| P2 | `viz/inspect_annotated.py` | 2h | annotate_multi_height.py（有数据） |
| P2 | `viz/compare_heights.py` | 2.5h | annotate_multi_height.py（有数据） |
| P3 | `viz/label_stats.py` | 1.5h | annotate_multi_height.py（有数据） |
| P3 | `preprocess_cache.py` 修改 | 1h | annotate_multi_height.py |
| P3 | `dataset.py` 修改 | 0.5h | preprocess_cache.py 修改 |

总估计：~16小时开发 + 测试时间（其中可视化工具 7.5h）。

**开发与验证交替节奏（推荐）：**
```
Day 1: collect（P1 采集工具）+ browse_raw_session（P1 可视化）→ 跑一次快速采集并人工确认
Day 2: annotate（P2 标注工具）+ inspect_annotated + compare_heights（P2 可视化）→ 验证标注质量
Day 3: label_stats（P3）+ preprocess_cache + dataset（P3 训练准备）→ 首次训练试跑
```

---

## 7. 关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| **Warp 参数来源** | 方案A：固定已知 pitch/yaw | §5.4 分析：Carla 真值精确，避免 rpyCalib 前 500 帧错误，34 组合无收敛代价 |
| **标注生成时机** | 离线（collect 后处理） | 简化采集流程（不需要 VisionIPC/modeld 进程），批量 GPU 处理更高效 |
| **标注格式** | canonical 字段直接存储（无 flat_label） | `PretrainedVisionModel` 返回 dict 而非 977-dim flat；decode_model_output() 解码为 canonical 格式（lane_lines (4,33,3) 等），与 extract_targets() 和 view_dual_data.py 绘图函数均兼容，preprocess_cache.py 无需新增提取函数 |
| **多高度同步方式** | 同一 Carla session 多相机 | §5.1：零额外仿真成本，时序完美对齐，H1 标注对应每帧精确场景 |
| **快速验证阶段** | H1+H6 仅 2 高度 | 验证 pipeline 端到端，最小资源投入，4 个相机 Carla 性能可接受 |
| **camera_height 存入缓存** | 是 | 为第二阶段显式高度注入（HeightConditionedHead）预留，无额外开销 |
| **log_sigma 处理** | 直接复制（不调整） | §5.5 决策：暂不引入额外超参，留作后续研究 |
| **质量过滤** | 采集后处理时过滤 | §5.4 Step 3 决策：采集时不过滤，标注后统计分析是否需要 |
| **车型** | 仅特斯拉 | §4.6 决策：暂时不考虑多种车型，减少变量 |
| **可视化工具基础** | 复用 visualizer.py + view_dual_data.py | 两文件已实现 lane/edge/lead 投影、BEV、置信度着色、键盘导航；新工具只增加多高度网格布局层 |
| **可视化工具位置** | `tools/dashcam/viz/` 子目录 | 与采集/训练工具分离；`__init__.py` 导出共用绘制函数供各工具复用 |
| **warp 后图像显示** | inspect/compare 工具实时计算 warp | 原始 road_rgb 存在 NPZ 中，warp 在工具里从 clip_info.json 参数重建；避免重复存储 warp 后图像（每帧省 ~600KB） |
