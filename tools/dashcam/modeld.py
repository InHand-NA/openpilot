#!/usr/bin/env python3
"""方案C: 修改 modeld warp 逻辑，wide-road-only 时 MEDMODEL 也使用 SBIGMODEL warp。

原理:
  当只有广角相机时，MEDMODEL 的标准 warp 使用 medmodel_fl=910，
  导致 ecam(567)/medmodel(910) = 0.62 上采样，像素信息严重不足。

  本模块将 get_warp_matrix 的 bigmodel_frame 参数强制设为 True，
  使 MEDMODEL 也使用 sbigmodel_fl=455，变为 ecam(567)/sbigmodel(455) = 1.25 降采样。

代价:
  MEDMODEL 的视野从 ~31° 扩大到 ~59°，不再匹配训练时的几何分布。
  模型可能输出异常。本模块仅用于实验验证。

用法:
  python -m openpilot.tools.dashcam.modeld
"""

# Monkey-patch get_warp_matrix BEFORE importing modeld (which binds a local reference at import time)
from openpilot.common.transformations import model as _model_mod
from openpilot.common.transformations.model import get_warp_matrix as _original_get_warp_matrix


def _patched_get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
  """Force SBIGMODEL warp (bigmodel_frame=True) for both MEDMODEL and SBIGMODEL inputs."""
  return _original_get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=True)


# Patch at module level — subsequent `from ... import get_warp_matrix` will get the patched version
_model_mod.get_warp_matrix = _patched_get_warp_matrix

# Now import modeld — its `from ... import get_warp_matrix` picks up our patch
from openpilot.selfdrive.modeld.modeld import main

if __name__ == "__main__":
  main()
