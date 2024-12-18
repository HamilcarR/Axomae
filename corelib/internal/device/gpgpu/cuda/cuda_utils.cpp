#include "cuda_utils.h"
#include <sstream>

std::size_t kernel_argpack_t::computeThreadsNumber() const {
  return (block_size.x * num_blocks.x) * (block_size.y * num_blocks.y) * (block_size.z * num_blocks.z);
}

std::size_t kernel_argpack_t::computeWarpNumber() const {
  std::size_t thread_num = computeThreadsNumber();
  return thread_num % AX_GPU_WARP_SIZE == 0 ? thread_num / AX_GPU_WARP_SIZE : (thread_num / AX_GPU_WARP_SIZE) + 1;
}

namespace ax_cuda::utils {

  /* Get device info . Throws GenericException*/
  std::string cuda_info_device() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
      CUDA_ERROR_CHECK(err);
      return "";
    }
    std::ostringstream os;
    for (int i = 0; i < device_count; i++) {
      cudaDeviceProp deviceProp{};
      cudaGetDeviceProperties(&deviceProp, i);
      // clang-format off
      os << "Device " << i << ": " << deviceProp.name << "\n";
      os << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
      os << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << "\n";
      os << "  Multiprocessors: " << deviceProp.multiProcessorCount << "\n";
      os << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << "\n";
      os << "  Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << "\n";
      os << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << "\n";
      // clang-format on
    }
    return os.str();
  }
}  // namespace ax_cuda::utils