#ifndef AXOMAE_KERNEL_LAUNCH_INTERFACE_H
#define AXOMAE_KERNEL_LAUNCH_INTERFACE_H

#include "device_utils.h"

#if defined(AXOMAE_USE_CUDA)
#include "cuda/cuda_utils.h"
#include "cuda/kernel_launch.cuh"

template<class F, class... Args>
void exec_kernel(const kernel_argpack_t &argpack, F &&kernel, Args &&...args) {
  KernelLauncher::launch(argpack.num_blocks , argpack.block_size , argpack.shared_mem_bytes , argpack.stream , kernel, std::forward<Args>(args)...);
}

#endif
#endif  // AXOMAE_KERNEL_LAUNCH_INTERFACE_H
