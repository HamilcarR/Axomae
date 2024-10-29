#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#if defined(AXOMAE_USE_CUDA)
#  include "cuda/cuda_utils.h"
/* TODO: AX_* to lowercase conforming to macros guideline*/
#  define AX_DEVICE_CALLABLE __host__ __device__
#  define AX_DEVICE_SHARED __shared__
#  define AX_DEVICE_ONLY __device__
#  define AX_DEVICE_CONST __device__ __constant__
#  define AX_HOST_ONLY __host__
#  define AX_KERNEL __global__

struct kernel_argpack_t {
  dim3 num_blocks;
  dim3 block_size;
  std::size_t shared_mem_bytes{};
  cudaStream_t stream{};
};

#else
#  define AX_DEVICE_CALLABLE
#  define AX_DEVICE_SHARED
#  define AX_DEVICE_ONLY
#  define AX_DEVICE_CONST
#  define AX_HOST_ONLY
#  define AX_KERNEL
#endif
#endif  // DEVICE_UTILS_H
