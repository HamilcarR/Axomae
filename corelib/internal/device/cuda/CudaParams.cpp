#include "CudaParams.h"

namespace ax_cuda {

  void CudaParams::setMemcpyKind(cudaMemcpyKind copy_kind) { memory_params.memcpy_kind = copy_kind; }
  void CudaParams::setChanDescriptors(int x, int y, int z, int a, cudaChannelFormatKind k) {
    chan_descriptor_params.format_desc = cudaCreateChannelDesc(x, y, z, a, k);
  }
  cudaChannelFormatDesc CudaParams::getChanDescriptors() const { return chan_descriptor_params.format_desc; }

}  // namespace ax_cuda