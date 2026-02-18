"""Lane line evaluation metrics: compare model output with ground truth.

Computes per-frame and cumulative statistics for lateral offset error,
detection rate, and lane width accuracy.
"""

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants

# Only evaluate ego lane boundaries (near-left=1, near-right=2)
_EVAL_INDICES = {1: 'near_left', 2: 'near_right'}


class LaneEvaluator:
  """Evaluate lane line detection accuracy against ground truth."""

  def __init__(self):
    self.x_idxs = np.array(ModelConstants.X_IDXS)
    self.history = []

    # Only evaluate X <= 100m
    self._eval_mask = self.x_idxs <= 100.0
    # Distance range masks for segmented evaluation (within 100m)
    self._near_mask = self.x_idxs <= 30.0
    self._mid_mask = (self.x_idxs > 30.0) & (self.x_idxs <= 60.0)
    self._far_mask = (self.x_idxs > 60.0) & (self.x_idxs <= 100.0)

  def evaluate(self, model_lines, model_probs, gt_lines, gt_probs):
    """Per-frame evaluation.

    Args:
      model_lines: list of 4 arrays, each 33x3 (from modelV2.laneLines).
      model_probs: list of 4 floats (from modelV2.laneLineProbs).
      gt_lines: list of 4 arrays, each 33x3 (from LaneGroundTruth).
      gt_probs: list of 4 floats (1.0 if GT lane exists, 0.0 otherwise).

    Returns:
      dict of evaluation metrics for this frame.
    """
    metrics = {}
    all_errors = []

    # Per-line lateral error (y-coordinate), ego lane only
    for i, name in _EVAL_INDICES.items():
      if gt_probs[i] < 0.5 or model_probs[i] < 0.5:
        continue

      y_model = model_lines[i][self._eval_mask, 1]
      y_gt = gt_lines[i][self._eval_mask, 1]
      errors = np.abs(y_model - y_gt)  # NaN in GT propagates as NaN

      for seg_name, mask in [('near', self._near_mask[self._eval_mask]),
                              ('mid', self._mid_mask[self._eval_mask]),
                              ('far', self._far_mask[self._eval_mask]),
                              ('all', np.ones(np.sum(self._eval_mask), dtype=bool))]:
        seg_errors = errors[mask]
        n_valid = np.sum(~np.isnan(seg_errors))
        if n_valid > 0:
          metrics[f'{name}_{seg_name}_mae'] = float(np.nanmean(seg_errors))
          metrics[f'{name}_{seg_name}_rmse'] = float(np.sqrt(np.nanmean(seg_errors ** 2)))

      valid_errors = errors[~np.isnan(errors)]
      all_errors.extend(valid_errors.tolist())

    # Overall MAE across ego lane lines
    if all_errors:
      arr = np.array(all_errors)
      metrics['overall_mae'] = float(np.mean(arr))
      metrics['overall_rmse'] = float(np.sqrt(np.mean(arr ** 2)))

    # Detection rate, ego lane only
    tp = fp = fn = 0
    for i in _EVAL_INDICES:
      gt_exists = gt_probs[i] > 0.5
      model_detected = model_probs[i] > 0.5
      if gt_exists and model_detected:
        tp += 1
      elif gt_exists and not model_detected:
        fn += 1
      elif not gt_exists and model_detected:
        fp += 1

    metrics['tp'] = tp
    metrics['fp'] = fp
    metrics['fn'] = fn
    if tp + fn > 0:
      metrics['recall'] = tp / (tp + fn)
    if tp + fp > 0:
      metrics['precision'] = tp / (tp + fp)

    # Lane width error (ego lane: near-left[1] and near-right[2]), within 100m
    if model_probs[1] > 0.5 and model_probs[2] > 0.5 and gt_probs[1] > 0.5 and gt_probs[2] > 0.5:
      m = self._eval_mask
      model_width = model_lines[2][m, 1] - model_lines[1][m, 1]
      gt_width = gt_lines[2][m, 1] - gt_lines[1][m, 1]
      width_err = np.abs(model_width - gt_width)
      if np.any(~np.isnan(width_err)):
        metrics['width_mae'] = float(np.nanmean(width_err))

    self.history.append(metrics)
    return metrics

  def get_summary(self):
    """Compute cumulative statistics across all evaluated frames.

    Returns:
      dict of metric_name -> mean value over all frames.
    """
    if not self.history:
      return {}

    all_keys = set()
    for m in self.history:
      all_keys.update(m.keys())

    summary = {}
    for key in sorted(all_keys):
      values = [m[key] for m in self.history if key in m]
      if values and isinstance(values[0], (int, float)):
        summary[key] = float(np.mean(values))

    return summary
