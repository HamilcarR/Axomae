#ifndef CU_MACRO_H
#define CU_MACRO_H

#include "Logger.h"
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#define AX_DEVICE_CALLABLE __host__ __device__
#define AX_DEVICE_SHARED __shared__
#define AX_DEVICE_ONLY __device__
#define AX_HOST_ONLY __host__
#define AX_KERNEL __global__

#define GPU_SYNCTHREAD __syncthreads()
#define GPU_SYNCWARP __syncwarp()

#define GPU_FASTCOS(value) __cosf(value)
#define GPU_FASTSIN(value) __sinf(value)
#define GPU_FASTEXP(value) __expf(value)

#define GPU_ATOMICADD(address, val) atomicAdd(address, val)
#define GPU_ATOMICSUB(address, val) atomicSub(address, val)
#define GPU_ATOMICMIN(address, val) atomicMin(address, val)
#define GPU_ATOMICMAX(address, val) atomicMax(address, val)

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
