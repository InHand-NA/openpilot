"""Loss functions for driving vision model training.

MDN outputs: raw tensor of shape (B, 2N) where first N = mu, last N = log(sigma).
BCE outputs: raw logits passed to BCEWithLogitsLoss.

Masking: lane_lines and road_edges losses are masked by both:
  - probability GT (prob > 0.5) at lane/edge level
  - per-point validity mask (NaN-free points from GT truncation at hill crests)
"""

import torch
import torch.nn as nn

from openpilot.tools.dashcam.train.config import ModelConfig, TrainConfig


class GaussianNLLLoss(nn.Module):
  """Gaussian negative log-likelihood for MDN outputs.

  Input raw (B, 2N): first N values are mu, last N are log(sigma).
  Target (B, N).
  Loss = 0.5 * ((target - mu) / sigma)^2 + log(sigma)
  """

  def __init__(self, min_log_sigma: float = -7.0, max_log_sigma: float = 7.0):
    super().__init__()
    self.min_log_sigma = min_log_sigma
    self.max_log_sigma = max_log_sigma

  def forward(self, raw: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    n = raw.shape[-1] // 2
    mu = raw[..., :n]
    log_sigma = torch.clamp(raw[..., n:], self.min_log_sigma, self.max_log_sigma)
    sigma = torch.exp(log_sigma)

    # (target - mu)^2 / (2 * sigma^2) + log(sigma)
    nll = 0.5 * ((target - mu) / sigma) ** 2 + log_sigma

    if mask is not None:
      # Expand mask to match nll shape
      while mask.dim() < nll.dim():
        mask = mask.unsqueeze(-1)
      mask = mask.expand_as(nll)
      count = mask.sum()
      if count > 0:
        return (nll * mask).sum() / count
      return nll.new_zeros(())  # no valid samples
    return nll.mean()


class DrivingLoss(nn.Module):
  """Combined loss for all output heads.

  lane_lines:      GaussianNLL, masked by (lane_lines_prob > 0.5) AND per-point valid
  lane_lines_prob: BCEWithLogitsLoss
  road_edges:      GaussianNLL, masked by (road_edges_prob > 0.5) AND per-point valid
  lead:            GaussianNLL, masked by lead_prob > 0.5
  lead_prob:       BCEWithLogitsLoss
  pose:            GaussianNLL (no mask)
  road_transform:  GaussianNLL (no mask)
  """

  def __init__(self, model_cfg: ModelConfig | None = None, train_cfg: TrainConfig | None = None):
    super().__init__()
    if model_cfg is None:
      model_cfg = ModelConfig()
    if train_cfg is None:
      train_cfg = TrainConfig()

    self.weights = train_cfg.loss_weights
    self.gnll = GaussianNLLLoss()
    self.bce = nn.BCEWithLogitsLoss(reduction='mean')

  def forward(self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}

    # --- lane_lines: MDN with prob mask AND per-point valid mask ---
    # pred: (B, 528) -> reshape to (B, 4, 33, 4) where 4 = 2mu + 2sigma
    # target: (B, 4, 33, 2)
    ll_pred = preds['lane_lines'].reshape(-1, 4, 33, 4)
    ll_target = targets['lane_lines']  # (B, 4, 33, 2)
    ll_prob_mask = (targets['lane_lines_prob'] > 0.5).float()  # (B, 4)
    ll_valid = targets['lane_lines_valid']  # (B, 4, 33)
    # Combined mask: lane must be active AND point must be valid
    ll_mask = ll_prob_mask.unsqueeze(-1) * ll_valid  # (B, 4, 33)
    losses['lane_lines'] = self.gnll(ll_pred, ll_target, ll_mask)

    # --- lane_lines_prob: BCE ---
    # pred: (B, 8) -> reshape (B, 4, 2) as logit pairs [1-p, p]
    # target: (B, 4) probs -> expand to (B, 4, 2) as [1-p, p]
    llp_pred = preds['lane_lines_prob']  # (B, 8)
    llp_target_raw = targets['lane_lines_prob']  # (B, 4)
    llp_target = torch.stack([1.0 - llp_target_raw, llp_target_raw], dim=-1)  # (B, 4, 2)
    losses['lane_lines_prob'] = self.bce(llp_pred.reshape(-1, 4, 2), llp_target)

    # --- road_edges: MDN with prob mask AND per-point valid mask ---
    # pred: (B, 264) -> reshape to (B, 2, 33, 4)
    # target: (B, 2, 33, 2)
    re_pred = preds['road_edges'].reshape(-1, 2, 33, 4)
    re_target = targets['road_edges']  # (B, 2, 33, 2)
    re_prob_mask = (targets['road_edges_prob'] > 0.5).float()  # (B, 2)
    re_valid = targets['road_edges_valid']  # (B, 2, 33)
    re_mask = re_prob_mask.unsqueeze(-1) * re_valid  # (B, 2, 33)
    losses['road_edges'] = self.gnll(re_pred, re_target, re_mask)

    # --- lead: MDN with mask ---
    # pred: (B, 144) -> reshape to (B, 3, 6, 8) where 8 = 4mu + 4sigma
    # target: (B, 3, 6, 4)
    lead_pred = preds['lead'].reshape(-1, 3, 6, 8)
    lead_target = targets['lead']  # (B, 3, 6, 4)
    lead_mask = (targets['lead_prob'] > 0.5).float()  # (B, 3)
    lead_mask = lead_mask.unsqueeze(-1).expand(-1, -1, 6)  # (B, 3, 6)
    losses['lead'] = self.gnll(lead_pred, lead_target, lead_mask)

    # --- lead_prob: BCE ---
    losses['lead_prob'] = self.bce(preds['lead_prob'], targets['lead_prob'])

    # --- pose: MDN, no mask ---
    # pred: (B, 12) -> 6mu + 6sigma, target: (B, 6)
    losses['pose'] = self.gnll(preds['pose'], targets['pose'])

    # --- road_transform: MDN, no mask ---
    losses['road_transform'] = self.gnll(preds['road_transform'], targets['road_transform'])

    # Weighted total
    total = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
    return total, losses
