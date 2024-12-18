#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#if defined(AXOMAE_USE_CUDA)
#  include "cuda/cuda_utils.h"
#  define ax_device_callable __host__ __device__
#  define ax_device_shared __shared__
#  define ax_device_only __device__
#  define ax_device_inlined __device__ inline
#  define ax_device_const __device__ __constant__
#  define ax_host_only __host__
#  define ax_kernel __global__

struct kernel_argpack_t {
  dim3 num_blocks{1, 1, 1};
  dim3 block_size{1, 1, 1};
  std::size_t shared_mem_bytes{};
  cudaStream_t stream{};

  std::size_t computeThreadsNumber() const { return (block_size.x * num_blocks.x) * (block_size.y * num_blocks.y) * (block_size.z * num_blocks.z); }
};

#else
#  define ax_device_callable
#  define ax_device_shared
#  define ax_device_only
#  define ax_device_const
#  define ax_host_only
#  define ax_kernel
#endif
#endif  // DEVICE_UTILS_H
