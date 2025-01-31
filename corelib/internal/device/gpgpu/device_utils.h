#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#if defined(AXOMAE_USE_CUDA)
#  include "cuda/cuda_utils.h"
#  define ax_device_callable __host__ __device__
#  define ax_device_shared __shared__
#  define ax_device_only __device__
#  define ax_device_inlined __device__ inline
#  define ax_device_callable_inlined __device__ __host__ inline
#  define ax_device_const __device__ __constant__
#  define ax_device_managed __managed__
#  define ax_host_only __host__
#  define ax_kernel __global__

#else
#  define ax_device_callable
#  define ax_device_shared
#  define ax_device_only
#  define ax_device_inlined
#  define ax_device_callable_inlined inline
#  define ax_device_const
#  define ax_device_managed
#  define ax_host_only
#  define ax_kernel
#endif
#endif  // DEVICE_UTILS_H
