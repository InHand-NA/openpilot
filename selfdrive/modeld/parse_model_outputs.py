import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants

def safe_exp(x, out=None):
  # -11 is around 10**14, more causes float16 overflow
  return np.exp(np.clip(x, -np.inf, 11), out=out)

def sigmoid(x):
  return 1. / (1. + safe_exp(-x))

def softmax(x, axis=-1):
  x -= np.max(x, axis=axis, keepdims=True)
  if x.dtype == np.float32 or x.dtype == np.float64:
    safe_exp(x, out=x)
  else:
    x = safe_exp(x)
  x /= np.sum(x, axis=axis, keepdims=True)
  return x

class Parser:
  def __init__(self, ignore_missing=False):
    self.ignore_missing = ignore_missing

  def check_missing(self, outs, name):
    missing = name not in outs
    if missing and not self.ignore_missing:
      raise ValueError(f"Missing output {name}")
    return missing

  def parse_categorical_crossentropy(self, name, outs, out_shape=None):
    if self.check_missing(outs, name):
      return
    raw = outs[name]
    if out_shape is not None:
      raw = raw.reshape((raw.shape[0],) + out_shape)
    outs[name] = softmax(raw, axis=-1)

  def parse_binary_crossentropy(self, name, outs):
    if self.check_missing(outs, name):
      return
    raw = outs[name]
    outs[name] = sigmoid(raw)

  def parse_mdn(self, name, outs, in_N=0, out_N=1, out_shape=None):
    if self.check_missing(outs, name):
      return
    raw = outs[name]
    raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))

    n_values = (raw.shape[2] - out_N)//2
    pred_mu = raw[:,:,:n_values]
    pred_std = safe_exp(raw[:,:,n_values: 2*n_values])

    if in_N > 1:
      weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
      for i in range(out_N):
        weights[:,:,i - out_N] = softmax(raw[:,:,i - out_N], axis=-1)

      if out_N == 1:
        for fidx in range(weights.shape[0]):
          idxs = np.argsort(weights[fidx][:,0])[::-1]
          weights[fidx] = weights[fidx][idxs]
          pred_mu[fidx] = pred_mu[fidx][idxs]
          pred_std[fidx] = pred_std[fidx][idxs]
      full_shape = tuple([raw.shape[0], in_N] + list(out_shape))
      outs[name + '_weights'] = weights
      outs[name + '_hypotheses'] = pred_mu.reshape(full_shape)
      outs[name + '_stds_hypotheses'] = pred_std.reshape(full_shape)

      pred_mu_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
      pred_std_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
      for fidx in range(weights.shape[0]):
        for hidx in range(out_N):
          idxs = np.argsort(weights[fidx,:,hidx])[::-1]
          pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
          pred_std_final[fidx, hidx] = pred_std[fidx, idxs[0]]
    else:
      pred_mu_final = pred_mu
      pred_std_final = pred_std

    if out_N > 1:
      final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
    else:
      final_shape = tuple([raw.shape[0],] + list(out_shape))
    outs[name] = pred_mu_final.reshape(final_shape)
    outs[name + '_stds'] = pred_std_final.reshape(final_shape)

  def is_mhp(self, outs, name, shape):
    if self.check_missing(outs, name):
      return False
    if outs[name].shape[1] == 2 * shape:
      return False
    return True

  def parse_vision_outputs(self, outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # 相机位姿向量（6 自由度），用于相机里程计/短期运动估计（通常包含平移 xyz 与欧拉角 rpy）。
    self.parse_mdn('pose', outs, in_N=0, out_N=0, out_shape=(ModelConstants.POSE_WIDTH,))
    # 设备坐标系到广角相机的欧拉角关系（rpy），用于多相机几何对齐。
    self.parse_mdn('wide_from_device_euler', outs, in_N=0, out_N=0, out_shape=(ModelConstants.WIDE_FROM_DEVICE_WIDTH,))
    # 路面坐标变换相关的 6 维参数（可理解为与路面几何/姿态估计相关的位姿量）。
    self.parse_mdn('road_transform', outs, in_N=0, out_N=0, out_shape=(ModelConstants.POSE_WIDTH,))
    # 4 条车道线（如左/右可见与虚/实分类由其他分支给出），沿前方距离序列的几何量。
    self.parse_mdn('lane_lines', outs, in_N=0, out_N=0, out_shape=(ModelConstants.NUM_LANE_LINES,ModelConstants.IDX_N,ModelConstants.LANE_LINES_WIDTH))
    # 左右路缘线的几何序列。
    self.parse_mdn('road_edges', outs, in_N=0, out_N=0, out_shape=(ModelConstants.NUM_ROAD_EDGES,ModelConstants.IDX_N,ModelConstants.LANE_LINES_WIDTH))
    # 每条车道线存在/可见性的概率。
    self.parse_binary_crossentropy('lane_lines_prob', outs)
    # 未来时域的驾驶意图分布（如直行、变道左/右、变道准备等）。
    self.parse_categorical_crossentropy('desire_pred', outs, out_shape=(ModelConstants.DESIRE_PRED_LEN,ModelConstants.DESIRE_PRED_WIDTH))
    # 一组“元事件”在未来各时间片发生的概率，如“油门/制动导致脱离”“方向盘接管”“不同等级急刹”等，以及灯光/踏板操作等。
    self.parse_binary_crossentropy('meta', outs)
    # 前车（领航车）存在的概率。
    self.parse_binary_crossentropy('lead_prob', outs)
    # 前车轨迹及状态的时间序列预测。维度含义（每个时间点的 4 维量，训练中常用约定）：
    # 典型为前向距离 x、横向偏移 y、相对速度 vx、相对加速度 ax（单位米/米每秒/米每二次方秒），具体以训练配置为准。
    lead_mhp = self.is_mhp(outs, 'lead', ModelConstants.LEAD_MHP_SELECTION * ModelConstants.LEAD_TRAJ_LEN * ModelConstants.LEAD_WIDTH)
    lead_in_N, lead_out_N = (ModelConstants.LEAD_MHP_N, ModelConstants.LEAD_MHP_SELECTION) if lead_mhp else (0, 0)
    lead_out_shape = (ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH) if lead_mhp else \
        (ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH)
    self.parse_mdn('lead', outs, in_N=lead_in_N, out_N=lead_out_N, out_shape=lead_out_shape)
    return outs

  def parse_policy_outputs(self, outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    plan_mhp = self.is_mhp(outs, 'plan',  ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH)
    plan_in_N, plan_out_N = (ModelConstants.PLAN_MHP_N, ModelConstants.PLAN_MHP_SELECTION) if plan_mhp else (0, 0)
    self.parse_mdn('plan', outs, in_N=plan_in_N, out_N=plan_out_N, out_shape=(ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
    self.parse_categorical_crossentropy('desire_state', outs, out_shape=(ModelConstants.DESIRE_PRED_WIDTH,))
    return outs

  def parse_outputs(self, outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    outs = self.parse_vision_outputs(outs)
    outs = self.parse_policy_outputs(outs)
    return outs
