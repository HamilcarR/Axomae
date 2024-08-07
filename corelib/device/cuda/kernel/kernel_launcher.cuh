#ifndef KERNEL_LAUNCHER_CUH
#define KERNEL_LAUNCHER_CUH
#include "../CudaParams.h"
#include "../cuda_utils.h"
namespace ax_cuda {

  template<class F, class... Args>
  void exec_kernel(const kernel_argpack_t &argpack, F &&func, Args &&...args) {
#ifdef __NVCC__
    func<<<argpack.num_blocks, argpack.block_size, argpack.shared_mem_bytes, argpack.stream>>>(std::forward<Args>(args)...);
#endif
  }

}  // namespace ax_cuda

#endif