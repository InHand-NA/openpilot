# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t

from msgq.visionipc.visionipc cimport cl_mem, cl_context as cl_context_t
from msgq.visionipc.visionipc_pyx cimport VisionBuf, CLContext as BaseCLContext
from .commonmodel cimport CL_DEVICE_TYPE_DEFAULT, cl_get_device_id, cl_create_context, cl_release_context
from .commonmodel cimport mat3, ModelFrame as cppModelFrame, DrivingModelFrame as cppDrivingModelFrame, MonitoringModelFrame as cppMonitoringModelFrame

# OpenCL API for direct buffer creation (used by prepare_from_yuv)
cdef extern from "CL/cl.h":
  ctypedef int cl_int
  ctypedef unsigned long long cl_mem_flags
  enum:
    _CL_MEM_READ_ONLY "CL_MEM_READ_ONLY"
    _CL_MEM_COPY_HOST_PTR "CL_MEM_COPY_HOST_PTR"
  cl_mem clCreateBuffer(cl_context_t context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret)
  cl_int clReleaseMemObject(cl_mem memobj)


cdef class CLContext(BaseCLContext):
  def __cinit__(self):
    self.device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT)
    self.context = cl_create_context(self.device_id)

  def __dealloc__(self):
    if self.context:
      cl_release_context(self.context)

cdef class CLMem:
  @staticmethod
  cdef create(void * cmem):
    mem = CLMem()
    mem.mem = <cl_mem*> cmem
    return mem

  @property
  def mem_address(self):
    return <uintptr_t>(self.mem)

def cl_from_visionbuf(VisionBuf buf):
  return CLMem.create(<void*>&buf.buf.buf_cl)


cdef class ModelFrame:
  cdef cppModelFrame * frame
  cdef int buf_size

  def __dealloc__(self):
    del self.frame

  def prepare(self, VisionBuf buf, float[:] projection):
    cdef mat3 cprojection
    memcpy(cprojection.v, &projection[0], 9*sizeof(float))
    cdef cl_mem * data
    data = self.frame.prepare(buf.buf.buf_cl, buf.width, buf.height, buf.stride, buf.uv_offset, cprojection)
    return CLMem.create(data)

  def prepare_from_yuv(self, CLContext context,
                       cnp.ndarray[cnp.uint8_t, ndim=1] yuv_data,
                       int width, int height, int stride, int uv_offset,
                       float[:] projection):
    """Prepare model input from raw NV12 data without VisionBuf/VisionIPC."""
    cdef mat3 cprojection
    memcpy(cprojection.v, &projection[0], 9 * sizeof(float))
    cdef cl_int err = 0
    cdef cl_mem yuv_cl = clCreateBuffer(
      context.context,
      <cl_mem_flags>(_CL_MEM_READ_ONLY | _CL_MEM_COPY_HOST_PTR),
      yuv_data.shape[0],
      <void*>&yuv_data[0], &err)
    if err != 0:
      raise RuntimeError(f"clCreateBuffer failed with error {err}")
    cdef cl_mem * data
    data = self.frame.prepare(yuv_cl, width, height, stride, uv_offset, cprojection)
    clReleaseMemObject(yuv_cl)
    return CLMem.create(data)

  def buffer_from_cl(self, CLMem in_frames):
    cdef unsigned char * data2
    data2 = self.frame.buffer_from_cl(in_frames.mem, self.buf_size)
    return np.asarray(<cnp.uint8_t[:self.buf_size]> data2)


cdef class DrivingModelFrame(ModelFrame):
  cdef cppDrivingModelFrame * _frame

  def __cinit__(self, CLContext context, int temporal_skip):
    self._frame = new cppDrivingModelFrame(context.device_id, context.context, temporal_skip)
    self.frame = <cppModelFrame*>(self._frame)
    self.buf_size = self._frame.buf_size

cdef class MonitoringModelFrame(ModelFrame):
  cdef cppMonitoringModelFrame * _frame

  def __cinit__(self, CLContext context):
    self._frame = new cppMonitoringModelFrame(context.device_id, context.context)
    self.frame = <cppModelFrame*>(self._frame)
    self.buf_size = self._frame.buf_size

