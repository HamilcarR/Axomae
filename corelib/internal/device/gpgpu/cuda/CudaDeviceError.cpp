#include "../DeviceError.h"
#include <cuda_runtime_api.h>
#include <internal/common/exception/GenericException.h>
#include <internal/debug/Logger.h>

void cuAssert(cudaError_t error, const char *file, int line, bool abort) {
  if (error != cudaSuccess) {
    const char *err_str = cudaGetErrorString(error);
    std::string err = "CUDA GPU assert , error : " + std::to_string(error) + " " + std::string(err_str);
    LOGS(err + "\n File: " + file + "\n Line: " + std::to_string(line));
    if (abort)
      exit(error);
  }
}