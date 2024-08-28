#ifndef KERNEL_LAUNCH_CUH
#define KERNEL_LAUNCH_CUH
#include "cuda_utils.h"



class KernelLauncher{
 public:
  template <class F , class ...Args>
  static void launch(dim3 num_blocks , dim3 block_size , std::size_t shared_mem_bytes, cudaStream_t stream , F &kernel , Args&&...args){
    kernel<<< num_blocks , block_size , shared_mem_bytes , stream >>>(std::forward<Args>(args)...);
  }
};



#endif