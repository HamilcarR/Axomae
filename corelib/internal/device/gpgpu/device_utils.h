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

/* Synchronization */
#  define ax_gpu_syncthread __syncthreads()
#  define ax_gpu_syncwarp __syncwarp()
#  define ax_gpu_memfence __threadfence()
#  define ax_gpu_memfence_block __threadfence_block()
#  define ax_gpu_memfence_system __threadfence_system()

/* Math operations */
#  define AX_GPU_FASTCOS(value) __cosf(value)
#  define AX_GPU_FASTSIN(value) __sinf(value)
#  define AX_GPU_FASTEXP(value) __expf(value)
#  define AX_GPU_FASTLOG(value) __logf(value)
#  define AX_GPU_FASTSQRT(value) __fsqrt_rn(value)
#  define AX_GPU_FASTPOW(x, y) __powf((x), (y))
#  define AX_GPU_FASTTANH(value) tanhf(value)
#  define AX_GPU_FLOORF(val) floorf(val)
#  define AX_GPU_ABS(val) abs(val)

#else
#  define ax_device_callable
#  define ax_device_shared
#  define ax_device_only
#  define ax_device_inlined inline
#  define ax_device_callable_inlined inline
#  define ax_device_const
#  define ax_device_managed
#  define ax_host_only
#  define ax_kernel

/* Synchronization */
#  define ax_gpu_syncthread
#  define ax_gpu_syncwarp
#  define ax_gpu_memfence
#  define ax_gpu_memfence_block
#  define ax_gpu_memfence_system

/* Math operations */
#  define AX_GPU_FASTCOS(value) cosf(value)
#  define AX_GPU_FASTSIN(value) sinf(value)
#  define AX_GPU_FASTEXP(value) expf(value)
#  define AX_GPU_FASTLOG(value) logf(value)
#  define AX_GPU_FASTSQRT(value) sqrtf(value)
#  define AX_GPU_FASTPOW(x, y) powf((x), (y))
#  define AX_GPU_FASTTANH(value) tanhf(value)
#  define AX_GPU_FLOORF(val) floorf(val)
#  define AX_GPU_ABS(val) fabsf(val)

#endif
#endif  // DEVICE_UTILS_H
