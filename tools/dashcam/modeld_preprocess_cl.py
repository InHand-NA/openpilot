#!/usr/bin/env python3
"""GPU preprocessing for openpilot modeld input using original OpenCL kernels.

Exactly replicates the three-stage pipeline used by openpilot's modeld:
  Stage 1  rgb_to_nv12.cl:  BGR(1928×1208×3) → NV12 (W*H + W*H/2 bytes)
  Stage 2  transform.cl:    NV12 Y+UV → warped Y(512×256) + U(256×128) + V(256×128)
  Stage 3  loadyuv.cl:      Y+U+V → 6-channel layout (6, 128, 256) uint8

Usage:
  from openpilot.tools.dashcam.modeld_preprocess_cl import ModeldInputPreprocessorCL

  prep = ModeldInputPreprocessorCL()
  yuv6ch = prep.process(bgr_image, warp_matrix)  # (6, 128, 256) uint8
  prep.close()

  # Context manager form
  with ModeldInputPreprocessorCL() as prep:
      yuv6ch = prep.process(bgr_image, warp_matrix)
"""

import numpy as np
from pathlib import Path


# Camera dimensions (openpilot standard)
W, H = 1928, 1208
RGB_STRIDE = W * 3          # bytes per row in BGR image
RGB_SIZE = W * H            # Y-plane size / offset to UV plane in NV12
NV12_SIZE = W * H * 3 // 2  # NV12 total: W*H (Y) + W*H/2 (interleaved UV)

# Model input dimensions
MODEL_W, MODEL_H = 512, 256
UV_W, UV_H = 256, 128
UV_SIZE = UV_W * UV_H       # 32768 bytes — one sub-channel of the 6-channel output
MODEL_FRAME_SIZE = UV_SIZE * 6  # 196608 bytes = (6, 128, 256) flattened

# Absolute paths to original openpilot OpenCL kernels
_ROOT = Path(__file__).parent.parent.parent  # tools/dashcam/../../ → openpilot root
_NV12_CL = _ROOT / 'tools' / 'sim' / 'rgb_to_nv12.cl'
_TRANSFORM_CL = _ROOT / 'selfdrive' / 'modeld' / 'transforms' / 'transform.cl'
_LOADYUV_CL = _ROOT / 'selfdrive' / 'modeld' / 'transforms' / 'loadyuv.cl'


def _transform_scale_buffer(M: np.ndarray, s: float) -> np.ndarray:
  """Scale a warp matrix for half-resolution UV processing.

  Python reimplementation of common/mat.h:transform_scale_buffer.
  Maps: in_pt = (transform(out_pt/s + 0.5) - 0.5) * s
  """
  T_out = np.array([[1.0 / s, 0.0, 0.5], [0.0, 1.0 / s, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
  T_in = np.array([[s, 0.0, -0.5 * s], [0.0, s, -0.5 * s], [0.0, 0.0, 1.0]], dtype=np.float64)
  return T_in @ M @ T_out


class ModeldInputPreprocessorCL:
  """GPU replica of openpilot modeld preprocessing using original OpenCL kernels.

  Exactly replicates the three-stage pipeline used by modeld:
    Stage 1 (rgb_to_nv12.cl):  BGR → NV12 on GPU
    Stage 2 (transform.cl):    NV12 → warped Y(512×256) + U(256×128) + V(256×128)
    Stage 3 (loadyuv.cl):      Y/U/V → 6-channel [y0,y1,y2,y3,U,V] output

  Using original kernel sources guarantees pixel-identical output to modeld,
  preserving annotation accuracy in the offline pipeline.

  Args:
    ctx: pyopencl.Context (optional — creates default platform context if None)
  """

  def __init__(self, ctx=None):
    try:
      import pyopencl as cl
    except ImportError as exc:
      raise ImportError(
        "pyopencl is required for GPU preprocessing. Install with: pip install pyopencl"
      ) from exc

    if ctx is None:
      ctx = cl.create_some_context(interactive=False)
    self._ctx = ctx
    self._queue = cl.CommandQueue(ctx)
    self._cl = cl

    prog_nv12 = self._compile_nv12(cl, ctx)
    prog_transform = self._compile_transform(cl, ctx)
    prog_loadyuv = self._compile_loadyuv(cl, ctx)

    # Pre-create kernel instances to avoid per-call RepeatedKernelRetrieval overhead
    self._k_rgb_to_nv12    = cl.Kernel(prog_nv12,      'rgb_to_nv12')
    self._k_warp           = cl.Kernel(prog_transform,  'warpPerspective')
    self._k_loadys         = cl.Kernel(prog_loadyuv,    'loadys')
    self._k_loaduv         = cl.Kernel(prog_loadyuv,    'loaduv')

    mf = cl.mem_flags
    # Persistent GPU buffers — allocated once, reused every process() call
    self._bgr_cl  = cl.Buffer(ctx, mf.READ_ONLY,   W * H * 3)          # input BGR
    self._nv12_cl = cl.Buffer(ctx, mf.READ_WRITE,  NV12_SIZE)           # NV12 intermediate
    self._y_cl    = cl.Buffer(ctx, mf.READ_WRITE,  MODEL_W * MODEL_H)   # warped Y (512×256)
    self._u_cl    = cl.Buffer(ctx, mf.READ_WRITE,  UV_W * UV_H)         # warped U (256×128)
    self._v_cl    = cl.Buffer(ctx, mf.READ_WRITE,  UV_W * UV_H)         # warped V (256×128)
    self._m_y_cl  = cl.Buffer(ctx, mf.READ_ONLY,   9 * 4)               # Y warp matrix (9 floats)
    self._m_uv_cl = cl.Buffer(ctx, mf.READ_ONLY,   9 * 4)               # UV warp matrix (9 floats)
    self._out_cl  = cl.Buffer(ctx, mf.WRITE_ONLY,  MODEL_FRAME_SIZE)    # 6-channel output

  # ------------------------------------------------------------------
  # Kernel compilation helpers
  # ------------------------------------------------------------------

  def _compile_nv12(self, cl, ctx):
    """Compile rgb_to_nv12.cl with required dimension defines."""
    src = _NV12_CL.read_text()
    defines = (
      f"#define WIDTH {W}\n"
      f"#define HEIGHT {H}\n"
      f"#define RGB_STRIDE {RGB_STRIDE}\n"
      f"#define RGB_SIZE {RGB_SIZE}\n"
    )
    return cl.Program(ctx, defines + src).build()

  def _compile_transform(self, cl, ctx):
    """Compile transform.cl (warpPerspective — no defines needed)."""
    return cl.Program(ctx, _TRANSFORM_CL.read_text()).build()

  def _compile_loadyuv(self, cl, ctx):
    """Compile loadyuv.cl with model frame dimension defines."""
    src = _LOADYUV_CL.read_text()
    defines = (
      f"#define TRANSFORMED_WIDTH {MODEL_W}\n"
      f"#define TRANSFORMED_HEIGHT {MODEL_H}\n"
    )
    return cl.Program(ctx, defines + src).build()

  # ------------------------------------------------------------------
  # Main API
  # ------------------------------------------------------------------

  def process(self, bgr: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
    """Preprocess a single BGR frame to modeld 6-channel input on GPU.

    Faithfully replicates modeld's DrivingModelFrame::prepare() pipeline:
      rgb_to_nv12 → warpPerspective(Y) → warpPerspective(U) → warpPerspective(V)
      → loadys → loaduv(U) → loaduv(V)

    Channel order matches commonmodel.cc/loadyuv.cc:
      out[0*UV_SIZE .. 1*UV_SIZE) = y0  (even rows, even cols of Y_warped)
      out[1*UV_SIZE .. 2*UV_SIZE) = y1  (odd  rows, even cols)
      out[2*UV_SIZE .. 3*UV_SIZE) = y2  (even rows, odd  cols)
      out[3*UV_SIZE .. 4*UV_SIZE) = y3  (odd  rows, odd  cols)
      out[4*UV_SIZE .. 5*UV_SIZE) = U   (256×128)
      out[5*UV_SIZE .. 6*UV_SIZE) = V   (256×128)

    Args:
      bgr: (1208, 1928, 3) uint8 in BGR channel order (Carla BGRA[:3])
      warp_matrix: (3, 3) float64 dst→src warp matrix from get_warp_matrix()

    Returns:
      (6, 128, 256) uint8 — [y0, y1, y2, y3, U, V]
    """
    cl = self._cl
    q = self._queue

    assert bgr.shape == (H, W, 3) and bgr.dtype == np.uint8, \
      f"Expected BGR (1208, 1928, 3) uint8, got {bgr.shape} {bgr.dtype}"

    # --- Upload inputs ---
    cl.enqueue_copy(q, self._bgr_cl, np.ascontiguousarray(bgr))

    M_y  = warp_matrix.astype(np.float32).flatten()
    M_uv = _transform_scale_buffer(warp_matrix, 0.5).astype(np.float32).flatten()
    cl.enqueue_copy(q, self._m_y_cl,  M_y)
    cl.enqueue_copy(q, self._m_uv_cl, M_uv)

    # --- Stage 1: BGR → NV12 ---
    # global_work_size = (W//4, H//4): each work-item processes a 4×4 block
    self._k_rgb_to_nv12.set_args(self._bgr_cl, self._nv12_cl)
    cl.enqueue_nd_range_kernel(q, self._k_rgb_to_nv12, (W // 4, H // 4), None)

    # --- Stage 2a: Warp Y (512×256) from NV12 Y-plane ---
    # Y-plane: stride=W (bytes/row), px_stride=1 (1 byte/pixel), offset=0
    self._k_warp.set_args(
      self._nv12_cl,
      np.int32(W), np.int32(1), np.int32(0), np.int32(H), np.int32(W),
      self._y_cl,
      np.int32(MODEL_W), np.int32(0), np.int32(MODEL_H), np.int32(MODEL_W),
      self._m_y_cl,
    )
    cl.enqueue_nd_range_kernel(q, self._k_warp, (MODEL_W, MODEL_H), None)

    # --- Stage 2b: Warp U (256×128) from NV12 UV-plane ---
    # NV12 UV-plane: interleaved [U0,V0,U1,V1,...], starts at byte W*H
    # stride=W (W bytes per UV row), px_stride=2 (skip V), offset=W*H
    self._k_warp.set_args(
      self._nv12_cl,
      np.int32(W), np.int32(2), np.int32(RGB_SIZE), np.int32(H // 2), np.int32(W // 2),
      self._u_cl,
      np.int32(UV_W), np.int32(0), np.int32(UV_H), np.int32(UV_W),
      self._m_uv_cl,
    )
    cl.enqueue_nd_range_kernel(q, self._k_warp, (UV_W, UV_H), None)

    # --- Stage 2c: Warp V (256×128) from NV12 UV-plane ---
    # Same as U but src_offset += 1 (V byte follows U in each interleaved pair)
    self._k_warp.set_args(
      self._nv12_cl,
      np.int32(W), np.int32(2), np.int32(RGB_SIZE + 1), np.int32(H // 2), np.int32(W // 2),
      self._v_cl,
      np.int32(UV_W), np.int32(0), np.int32(UV_H), np.int32(UV_W),
      self._m_uv_cl,
    )
    cl.enqueue_nd_range_kernel(q, self._k_warp, (UV_W, UV_H), None)

    # --- Stage 3a: loadys — Y(512×256) → [y0,y1,y2,y3] at out[0..4*UV_SIZE) ---
    # global size = MODEL_W * MODEL_H / 8 = 16384  (each item handles 8 Y pixels)
    self._k_loadys.set_args(self._y_cl, self._out_cl, np.int32(0))
    cl.enqueue_nd_range_kernel(q, self._k_loadys, (MODEL_W * MODEL_H // 8,), None)

    # --- Stage 3b: loaduv — U(256×128) → out[4*UV_SIZE..5*UV_SIZE) ---
    self._k_loaduv.set_args(self._u_cl, self._out_cl, np.int32(UV_SIZE * 4))
    cl.enqueue_nd_range_kernel(q, self._k_loaduv, (UV_W * UV_H // 8,), None)

    # --- Stage 3c: loaduv — V(256×128) → out[5*UV_SIZE..6*UV_SIZE) ---
    self._k_loaduv.set_args(self._v_cl, self._out_cl, np.int32(UV_SIZE * 5))
    cl.enqueue_nd_range_kernel(q, self._k_loaduv, (UV_W * UV_H // 8,), None)

    # --- Download result ---
    out = np.empty(MODEL_FRAME_SIZE, dtype=np.uint8)
    cl.enqueue_copy(q, out, self._out_cl)
    q.finish()

    return out.reshape(6, UV_H, UV_W)  # (6, 128, 256)

  def close(self):
    """Release all OpenCL GPU buffers."""
    for buf in (self._bgr_cl, self._nv12_cl, self._y_cl, self._u_cl, self._v_cl,
                self._m_y_cl, self._m_uv_cl, self._out_cl):
      try:
        buf.release()
      except Exception:
        pass

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.close()
