"""Pretrained driving vision model from openpilot ONNX weights.

Loads the openpilot driving_vision.onnx model (23M params, trained on millions of
real-world driving miles) and wraps it as a PyTorch module for fine-tuning.

The original ONNX outputs 1576 dims as a flat concatenation of 12 components.
This wrapper slices out the 977 useful dims and returns them as a dict.

Output mapping (from original ONNX Concat order):
  [0:55]    meta          — skip (truck-specific)
  [55:87]   desire_pred   — skip (ACC/ALC-specific)
  [87:99]   pose          — keep (12-dim MDN: 6 mu + 6 sigma)
  [99:105]  wide_from_device_euler — keep (6-dim MDN: 3 mu + 3 sigma)
  [105:117] road_transform — keep (12-dim MDN)
  [117:645] lane_lines    — keep (528-dim MDN)
  [645:653] lane_lines_prob — keep (8-dim logits)
  [653:917] road_edges    — keep (264-dim MDN)
  [917:1061] lead         — keep (144-dim MDN)
  [1061:1064] lead_prob   — keep (3-dim logits)
  [1064:1576] summarizer  — skip (512-dim feature vector)

Inputs:
  img:     (B, 12, 128, 256) uint8 — road camera (2 frames x 6ch YUV420)
  big_img: (B, 12, 128, 256) uint8 — wide camera (2 frames x 6ch YUV420)

Usage:
  model = PretrainedVisionModel('selfdrive/modeld/models/driving_vision.onnx')
  model.eval()
  out = model(img, big_img)  # dict of named tensors, float32
"""

import torch
import torch.nn as nn

try:
  import onnx2torch
except ImportError as e:
  raise ImportError("onnx2torch is required: pip install onnx2torch") from e


# Slices into the 1576-dim flat ONNX output
ONNX_OUTPUT_SLICES = {
  'pose':                   slice(87,   99),   # 12 dims (6 mu + 6 sigma)
  'wide_from_device_euler': slice(99,   105),  # 6 dims  (3 mu + 3 sigma)
  'road_transform':         slice(105,  117),  # 12 dims
  'lane_lines':             slice(117,  645),  # 528 dims
  'lane_lines_prob':        slice(645,  653),  # 8 dims
  'road_edges':             slice(653,  917),  # 264 dims
  'lead':                   slice(917,  1061), # 144 dims
  'lead_prob':              slice(1061, 1064), # 3 dims
}

# Names used by the original model's output dict (matches train.py / losses.py)
OUTPUT_NAMES = list(ONNX_OUTPUT_SLICES.keys())


class PretrainedVisionModel(nn.Module):
  """Openpilot driving vision model loaded from ONNX, wrapped for PyTorch fine-tuning.

  The backbone (all onnx2torch-converted weights) is frozen by default.
  Use freeze_backbone() / unfreeze_all() for staged fine-tuning.

  Args:
    onnx_path: path to driving_vision.onnx
    freeze: if True (default), freeze all backbone weights on init
  """

  def __init__(self, onnx_path: str, freeze: bool = True):
    super().__init__()
    self._backbone = onnx2torch.convert(onnx_path)
    self._backbone.eval()
    if freeze:
      self.freeze_backbone()

  def freeze_backbone(self):
    """Freeze all backbone parameters (inference-only mode)."""
    for p in self._backbone.parameters():
      p.requires_grad_(False)

  def unfreeze_all(self):
    """Unfreeze all backbone parameters for full fine-tuning."""
    for p in self._backbone.parameters():
      p.requires_grad_(True)

  def forward(self, img: torch.Tensor, big_img: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run model on dual-camera inputs.

    Args:
      img:     (B, 12, 128, 256) uint8 — road camera YUV420
      big_img: (B, 12, 128, 256) uint8 — wide camera YUV420

    Returns:
      dict with keys: pose, wide_from_device_euler, road_transform,
                      lane_lines, lane_lines_prob, road_edges, lead, lead_prob
      All values are float32 tensors.

    Note:
      The underlying ONNX model has a fixed batch size of 1. For B > 1,
      samples are processed sequentially and results are stacked. Use
      gradient accumulation (accumulate_grad_batches) to simulate large batches.
    """
    B = img.shape[0]
    if B == 1:
      flat = self._backbone(img, big_img).float()  # (1, 1576)
    else:
      # Process each sample individually; ONNX model is fixed to batch=1
      flats = [self._backbone(img[i:i+1], big_img[i:i+1]).float() for i in range(B)]
      flat = torch.cat(flats, dim=0)  # (B, 1576)
    return {name: flat[:, sl] for name, sl in ONNX_OUTPUT_SLICES.items()}

  def n_trainable_params(self) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def n_total_params(self) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in self.parameters())
