#ifndef CU_MACRO_H
#define CU_MACRO_H

#include "cuda_macros.h"
#include "internal/debug/Logger.h"
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime_api.h>

#define CUDA_ERROR_CHECK(ans) \
  { \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    const char *err_str = cudaGetErrorString(code);
    std::string err = "CUDA GPU assert :" + std::string(err_str);
    LOGS(err + "\n File: " + file + "\n Line: " + std::to_string(line));
    if (abort)
      exit(code);
  }
}

namespace ax_cuda::utils {
  std::string cuda_info_device();
}  // namespace ax_cuda::utils
#endif  // CU_MACRO_H
