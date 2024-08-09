#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#include "cuda/cuda_utils.h"

struct kernel_argpack_t {
  dim3 num_blocks;
  dim3 block_size;
  std::size_t shared_mem_bytes{};
#if defined(__NVCC__)
  cudaStream_t stream{};
#elif defined(__HIPCC__)
  int stream{};
#endif
};

template<class F, class... Args>
void exec_kernel(const kernel_argpack_t &argpack, F &&func, Args &&...args) {
#if defined(__NVCC__)
  func<<<argpack.num_blocks, argpack.block_size, argpack.shared_mem_bytes, argpack.stream>>>(std::forward<Args>(args)...);
#endif
}

#endif  // DEVICE_UTILS_H
