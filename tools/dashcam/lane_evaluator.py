"""Lane line evaluation metrics: compare model output with ground truth.

Computes per-frame and cumulative statistics for lateral offset error,
detection rate (prob-based and position-based), and lane width accuracy.
"""

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants

# Only evaluate ego lane boundaries (near-left=1, near-right=2)
_EVAL_INDICES = {1: 'near_left', 2: 'near_right'}

# Position accuracy thresholds (meters) for per-point accuracy rate
_ACC_THRESHOLDS = [0.3, 0.5, 1.0]

# Position-based TP: line is accurate if >80% of valid points in <60m have error < 0.5m
_POS_TP_THRESHOLD = 0.5   # lateral error threshold in meters
_POS_TP_MIN_RATE = 0.80   # minimum accuracy rate to count as position-TP


class LaneEvaluator:
  """Evaluate lane line detection accuracy against ground truth."""

  def __init__(self):
    self.x_idxs = np.array(ModelConstants.X_IDXS)
    self.history = []

    # Only evaluate X <= 100m
    self._eval_mask = self.x_idxs <= 100.0
    # <60m mask for position accuracy focus
    self._close_mask = self.x_idxs <= 60.0
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
    pos_tp = pos_fp = pos_fn = 0
    for i, name in _EVAL_INDICES.items():
      gt_exists = gt_probs[i] > 0.5
      model_detected = model_probs[i] > 0.5

      if not gt_exists and not model_detected:
        continue

      if gt_exists and model_detected:
        y_model = model_lines[i][self._eval_mask, 1]
        y_gt = gt_lines[i][self._eval_mask, 1]
        errors = np.abs(y_model - y_gt)

        # Segmented MAE/RMSE (0-100m)
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

        # Per-point accuracy rates at multiple thresholds (<60m)
        close_errors = np.abs(model_lines[i][self._close_mask, 1] - gt_lines[i][self._close_mask, 1])
        close_valid = close_errors[~np.isnan(close_errors)]
        if len(close_valid) > 0:
          for thresh in _ACC_THRESHOLDS:
            rate = float(np.mean(close_valid < thresh))
            metrics[f'{name}_acc_{thresh:.1f}m'] = rate

          # Position-based TP: accurate if enough points within threshold in <60m
          pos_rate = float(np.mean(close_valid < _POS_TP_THRESHOLD))
          if pos_rate >= _POS_TP_MIN_RATE:
            pos_tp += 1
          else:
            pos_fp += 1  # detected but inaccurate
        else:
          pos_tp += 1  # no valid points to evaluate, count as TP

      elif gt_exists and not model_detected:
        pos_fn += 1
      elif not gt_exists and model_detected:
        pos_fp += 1

    # Overall MAE across ego lane lines
    if all_errors:
      arr = np.array(all_errors)
      metrics['overall_mae'] = float(np.mean(arr))
      metrics['overall_rmse'] = float(np.sqrt(np.mean(arr ** 2)))

    # Prob-based detection rate (original)
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

    # Position-based detection rate
    metrics['pos_tp'] = pos_tp
    metrics['pos_fp'] = pos_fp
    metrics['pos_fn'] = pos_fn

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

    # Count keys: sum instead of average
    _count_keys = {'tp', 'fp', 'fn', 'pos_tp', 'pos_fp', 'pos_fn'}

    # Prob-based detection counts
    total_tp = sum(m.get('tp', 0) for m in self.history)
    total_fp = sum(m.get('fp', 0) for m in self.history)
    total_fn = sum(m.get('fn', 0) for m in self.history)

    # Position-based detection counts
    total_pos_tp = sum(m.get('pos_tp', 0) for m in self.history)
    total_pos_fp = sum(m.get('pos_fp', 0) for m in self.history)
    total_pos_fn = sum(m.get('pos_fn', 0) for m in self.history)

    summary = {}
    for key in sorted(all_keys):
      if key in _count_keys or key in ('recall', 'precision'):
        continue
      values = [m[key] for m in self.history if key in m]
      if values and isinstance(values[0], (int, float)):
        summary[key] = float(np.mean(values))

    # Prob-based detection statistics
    summary['tp'] = total_tp
    summary['fp'] = total_fp
    summary['fn'] = total_fn
    total = total_tp + total_fp + total_fn
    if total > 0:
      summary['tp_rate'] = float(total_tp / total)
      summary['fp_rate'] = float(total_fp / total)
      summary['fn_rate'] = float(total_fn / total)
    if total_tp + total_fn > 0:
      summary['recall'] = float(total_tp / (total_tp + total_fn))
    if total_tp + total_fp > 0:
      summary['precision'] = float(total_tp / (total_tp + total_fp))

    # Position-based detection statistics (<60m, threshold=0.5m, min_rate=80%)
    summary['pos_tp'] = total_pos_tp
    summary['pos_fp'] = total_pos_fp
    summary['pos_fn'] = total_pos_fn
    total_pos = total_pos_tp + total_pos_fp + total_pos_fn
    if total_pos > 0:
      summary['pos_tp_rate'] = float(total_pos_tp / total_pos)
      summary['pos_fp_rate'] = float(total_pos_fp / total_pos)
      summary['pos_fn_rate'] = float(total_pos_fn / total_pos)
    if total_pos_tp + total_pos_fn > 0:
      summary['pos_recall'] = float(total_pos_tp / (total_pos_tp + total_pos_fn))
    if total_pos_tp + total_pos_fp > 0:
      summary['pos_precision'] = float(total_pos_tp / (total_pos_tp + total_pos_fp))

    return summary
