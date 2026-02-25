from dataclasses import dataclass, field


@dataclass
class ModelConfig:
  in_channels: int = 12  # 2 frames x 6 YUV420 channels
  stem_channels: int = 64
  stage_channels: tuple = (64, 128, 256, 512)
  stage_blocks: tuple = (2, 2, 6, 2)  # FastViT-S12
  mlp_ratio: float = 3.0
  backbone_out_dim: int = 1024  # final DWConv output
  feature_dim: int = 2048  # GAP -> FC output
  bottleneck_dim: int = 512
  use_relu: bool = False  # V1 single-cam default GELU (backward compatible)
  uint8_input: bool = False  # V1 single-cam default float32 input (backward compatible)

  # Bottleneck Policy head (L2Norm): {name: (raw_dim, hidden_dim, loss_type)}
  head1_outputs: dict = field(
    default_factory=lambda: {
      'lane_lines': (528, 64, 'mdn'),
      'lane_lines_prob': (8, 16, 'bce'),
      'road_edges': (264, 32, 'mdn'),
      'lead': (144, 64, 'mdn'),
      'lead_prob': (3, 16, 'bce'),
    }
  )

  # No-Bottleneck Policy head: {name: (raw_dim, hidden_dim, loss_type)}
  head2_outputs: dict = field(
    default_factory=lambda: {
      'pose': (12, 32, 'mdn'),
      'road_transform': (12, 32, 'mdn'),
    }
  )


@dataclass
class DualCameraModelConfig(ModelConfig):
  in_channels: int = 24  # concat: 2 cameras x 2 frames x 6ch
  use_relu: bool = True  # match pretrained (ReLU not GELU)
  uint8_input: bool = True  # model accepts uint8, internal normalization (match openpilot)


@dataclass
class TrainConfig:
  batch_size: int = 16
  epochs: int = 100
  lr: float = 1e-3
  weight_decay: float = 1e-4
  warmup_epochs: int = 5
  num_workers: int = 4
  val_split: float = 0.05
  save_every: int = 5
  log_every: int = 50
  grad_clip: float = 1.0

  loss_weights: dict = field(
    default_factory=lambda: {
      'lane_lines': 1.0,
      'lane_lines_prob': 1.0,
      'road_edges': 1.0,
      'lead': 0.5,
      'lead_prob': 1.0,
      'pose': 0.2,
      'road_transform': 0.2,
    }
  )
