"""Single-camera driving vision model with FastViT backbone.

Architecture:
  Input: (B, 12, 128, 256) — 2 frames x 6ch YUV420
  Backbone: FastViT RepMixer [2,2,6,2], channels [64,128,256,512]
  Output heads:
    - Bottleneck (L2Norm): lane_lines, lane_lines_prob, road_edges, lead, lead_prob
    - No-Bottleneck: pose, road_transform
  Total output: 971 dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpilot.tools.dashcam.train.config import ModelConfig


# ---------- Backbone building blocks ----------


class RepConv(nn.Module):
  """Reparameterizable depthwise conv: DWConv(3x3) + BN + Identity.

  During training, keeps separate branches for better gradient flow.
  At inference, fuse into a single DWConv via `fuse()`.
  """

  def __init__(self, channels: int):
    super().__init__()
    self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
    self.bn_dw = nn.BatchNorm2d(channels)
    self.bn_id = nn.BatchNorm2d(channels)
    self._fused = False

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._fused:
      return self.fused_conv(x)
    return self.bn_dw(self.dw(x)) + self.bn_id(x)

  def fuse(self):
    """Fuse parallel branches into a single depthwise conv for inference."""
    if self._fused:
      return
    # DWConv + BN branch
    k_dw, b_dw = self._fuse_bn(self.dw, self.bn_dw)
    # Identity + BN branch → equivalent to 3x3 identity kernel + BN
    k_id = torch.zeros_like(k_dw)
    k_id[:, :, 1, 1] = 1.0  # center pixel = identity for depthwise
    k_bn_id, b_bn_id = self._fuse_bn_to_conv(self.bn_id, k_id)
    # Merge
    fused_weight = k_dw + k_bn_id
    fused_bias = b_dw + b_bn_id
    conv = nn.Conv2d(
      fused_weight.shape[0],
      fused_weight.shape[0],
      3,
      padding=1,
      groups=fused_weight.shape[0],
      bias=True,
    )
    conv.weight.data = fused_weight
    conv.bias.data = fused_bias
    self.fused_conv = conv
    self._fused = True
    # Remove old params to save memory
    del self.dw, self.bn_dw, self.bn_id

  @staticmethod
  def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    w = conv.weight
    mean, var, gamma, beta = bn.running_mean, bn.running_var, bn.weight, bn.bias
    std = torch.sqrt(var + bn.eps)
    # Reshape for depthwise: (C, 1, kH, kW)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_w = w * scale
    fused_b = beta - mean * gamma / std
    return fused_w, fused_b

  @staticmethod
  def _fuse_bn_to_conv(bn: nn.BatchNorm2d, kernel: torch.Tensor):
    mean, var, gamma, beta = bn.running_mean, bn.running_var, bn.weight, bn.bias
    std = torch.sqrt(var + bn.eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_w = kernel * scale
    fused_b = beta - mean * gamma / std
    return fused_w, fused_b


class ConvFFN(nn.Module):
  """Convolutional feed-forward network: DWConv(7x7) -> 1x1(C->mlp_C) -> GELU -> 1x1(mlp_C->C)."""

  def __init__(self, channels: int, mlp_ratio: float = 3.0):
    super().__init__()
    hidden = int(channels * mlp_ratio)
    self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
    self.bn = nn.BatchNorm2d(channels)
    self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
    self.act = nn.GELU()
    self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.bn(self.dw(x))
    x = self.fc2(self.act(self.fc1(x)))
    return x


class RepMixerBlock(nn.Module):
  """Single RepMixer block: RepConv + ConvFFN, each with layer_scale and residual."""

  def __init__(self, channels: int, mlp_ratio: float = 3.0, layer_scale_init: float = 1e-5):
    super().__init__()
    self.token_mixer = RepConv(channels)
    self.ffn = ConvFFN(channels, mlp_ratio)
    self.ls1 = nn.Parameter(layer_scale_init * torch.ones(channels, 1, 1))
    self.ls2 = nn.Parameter(layer_scale_init * torch.ones(channels, 1, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.ls1 * self.token_mixer(x)
    x = x + self.ls2 * self.ffn(x)
    return x


class Downsample(nn.Module):
  """Spatial downsampling: DWConv(7x7, stride=2) + BN + Conv1x1 + BN."""

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.dw = nn.Conv2d(in_channels, in_channels, 7, stride=2, padding=3, groups=in_channels, bias=False)
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.bn2(self.pw(self.bn1(self.dw(x))))


class FastViTStage(nn.Module):
  """One stage of FastViT: optional Downsample + N x RepMixerBlock."""

  def __init__(self, in_channels: int, out_channels: int, num_blocks: int, mlp_ratio: float = 3.0, downsample: bool = True):
    super().__init__()
    self.downsample = Downsample(in_channels, out_channels) if downsample else None
    self.blocks = nn.Sequential(*[RepMixerBlock(out_channels, mlp_ratio) for _ in range(num_blocks)])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.downsample is not None:
      x = self.downsample(x)
    return self.blocks(x)


class SEBlock(nn.Module):
  """Squeeze-and-Excitation block: GAP -> FC(C->C//r) -> ReLU -> FC(C//r->C) -> Sigmoid."""

  def __init__(self, channels: int, reduction: int = 16):
    super().__init__()
    mid = max(channels // reduction, 1)
    self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
    self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    s = F.adaptive_avg_pool2d(x, 1)
    s = F.relu(self.fc1(s))
    s = torch.sigmoid(self.fc2(s))
    return x * s


class FastViTBackbone(nn.Module):
  """FastViT backbone: Stem + 4 Stages + DWConv expansion + SE + GAP + FC.

  Input:  (B, 12, 128, 256)
  Output: (B, feature_dim)
  """

  def __init__(self, cfg: ModelConfig):
    super().__init__()
    ch = cfg.stem_channels

    # Stem: Conv(in->64, s=2) + BN + GELU, DWConv(64, s=2) + BN + GELU, Conv1x1(64->64) + BN + GELU
    self.stem = nn.Sequential(
      nn.Conv2d(cfg.in_channels, ch, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ch),
      nn.GELU(),
      nn.Conv2d(ch, ch, 3, stride=2, padding=1, groups=ch, bias=False),
      nn.BatchNorm2d(ch),
      nn.GELU(),
      nn.Conv2d(ch, ch, 1, bias=False),
      nn.BatchNorm2d(ch),
      nn.GELU(),
    )

    # 4 stages
    stages = []
    in_ch = ch
    for i, (out_ch, n_blocks) in enumerate(zip(cfg.stage_channels, cfg.stage_blocks, strict=True)):
      downsample = i > 0  # Stage 0: no downsample (stem already did 4x)
      stages.append(FastViTStage(in_ch, out_ch, n_blocks, cfg.mlp_ratio, downsample))
      in_ch = out_ch
    self.stages = nn.Sequential(*stages)

    # Final: DWConv(512->1024) + SE + GELU + GAP + FC(1024->2048)
    final_ch = cfg.stage_channels[-1]
    self.final_dw = nn.Conv2d(final_ch, cfg.backbone_out_dim, 1, bias=False)
    self.final_bn = nn.BatchNorm2d(cfg.backbone_out_dim)
    self.se = SEBlock(cfg.backbone_out_dim)
    self.act = nn.GELU()
    self.fc = nn.Linear(cfg.backbone_out_dim, cfg.feature_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.stem(x)  # (B, 64, 32, 64)
    x = self.stages(x)  # (B, 512, 4, 8)
    x = self.act(self.se(self.final_bn(self.final_dw(x))))  # (B, 1024, 4, 8)
    x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, 1024)
    x = self.fc(x)  # (B, 2048)
    return x


# ---------- Head building blocks ----------


class ResBlock(nn.Module):
  """Residual FC block: FC(d->2d) -> ReLU -> FC(2d->d) + residual."""

  def __init__(self, dim: int):
    super().__init__()
    self.fc1 = nn.Linear(dim, dim * 2)
    self.fc2 = nn.Linear(dim * 2, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.fc2(F.relu(self.fc1(x)))


class OutputHead(nn.Module):
  """Single output head: FC(bottleneck_dim -> hidden) -> ReLU -> ResBlock -> FC(hidden -> out_dim)."""

  def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.ReLU(),
      ResBlock(hidden_dim),
      nn.Linear(hidden_dim, out_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


# ---------- Full model ----------


class DrivingVisionModel(nn.Module):
  """Single-camera driving vision model.

  Architecture:
    backbone -> feature (B, 2048)
    head1 (bottleneck, L2-normalized): lane_lines, lane_lines_prob, road_edges, lead, lead_prob
    head2 (no bottleneck): pose, road_transform

  Args:
    cfg: ModelConfig with architecture hyperparameters
  """

  def __init__(self, cfg: ModelConfig | None = None):
    super().__init__()
    if cfg is None:
      cfg = ModelConfig()
    self.cfg = cfg

    self.backbone = FastViTBackbone(cfg)

    bd = cfg.bottleneck_dim

    # Head 1: Bottleneck Policy with L2Norm
    self.head1_summarizer = nn.Sequential(
      nn.Linear(cfg.feature_dim, bd),
      ResBlock(bd),
      ResBlock(bd),
      nn.Linear(bd, bd),
    )
    self.head1_hydra = nn.Sequential(
      ResBlock(bd),
      ResBlock(bd),
    )
    self.head1_outputs = nn.ModuleDict({name: OutputHead(bd, hidden, out_dim) for name, (out_dim, hidden, _) in cfg.head1_outputs.items()})

    # Head 2: No-Bottleneck Policy
    self.head2_summarizer = nn.Sequential(
      nn.Linear(cfg.feature_dim, bd),
      ResBlock(bd),
      ResBlock(bd),
    )
    self.head2_hydra = nn.Sequential(
      ResBlock(bd),
      ResBlock(bd),
    )
    self.head2_outputs = nn.ModuleDict({name: OutputHead(bd, hidden, out_dim) for name, (out_dim, hidden, _) in cfg.head2_outputs.items()})

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    feat = self.backbone(x)  # (B, 2048)

    # Head 1: Bottleneck with L2 normalization
    h1 = self.head1_summarizer(feat)
    h1 = F.normalize(h1, p=2, dim=-1)
    h1 = self.head1_hydra(h1)  # (B, 512)

    # Head 2: No bottleneck
    h2 = self.head2_summarizer(feat)
    h2 = self.head2_hydra(h2)  # (B, 512)

    outputs: dict[str, torch.Tensor] = {}
    for name, head in self.head1_outputs.items():
      outputs[name] = head(h1)
    for name, head in self.head2_outputs.items():
      outputs[name] = head(h2)
    return outputs

  def fuse_repconv(self):
    """Fuse all RepConv blocks for inference (reparameterization)."""
    for m in self.modules():
      if isinstance(m, RepConv):
        m.fuse()
