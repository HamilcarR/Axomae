#ifndef DEVICEERROR_H
#define DEVICEERROR_H
#include "cuda/cuda_utils.h"
#include "project_macros.h"
#include <cstdint>

class DeviceError {
 public:
  enum DEVICETYPE : uint { CUDA = 0, OPENCL = 1, HIP = 2 };

 private:
  uint64_t err;

 public:
  CLASS_CM(DeviceError)
  explicit DeviceError(cudaError_t error);
  [[nodiscard]] bool check(DEVICETYPE type) const;
  [[nodiscard]] decltype(auto) cast_error(DEVICETYPE T) const {
    if (T == CUDA) {
      return static_cast<cudaError_t>(err);
    }
    AX_UNREACHABLE
    return static_cast<cudaError_t>(err);
  }
};
#define AXCUDA_ERROR_CHECK(ans) \
  { \
    cuAssert((ans), __FILE__, __LINE__); \
  }
inline void cuAssert(const DeviceError &err, const char *file, int line, bool abort = false) {
  cudaError_t code = err.cast_error(DeviceError::CUDA);
  if (code != cudaSuccess) {
    const char *err_str = cudaGetErrorString(code);
    std::string err = "CUDA GPU assert , error : " + std::to_string(code) + " " + std::string(err_str);
    LOGS(err + "\n File: " + file + "\n Line: " + std::to_string(line));
    if (abort)
      exit(code);
  }
}

#endif  // DEVICEERROR_H
