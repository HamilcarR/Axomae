#ifndef DEVICEERROR_H
#define DEVICEERROR_H
#include "internal/macro/project_macros.h"
#include <cstdint>
#include <driver_types.h>

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
#if defined(AXOMAE_USE_CUDA)

#  define DEVICE_ERROR_CHECK(ans) \
    { \
      cuAssert((ans), __FILE__, __LINE__); \
    }
void cuAssert(const DeviceError &err, const char *file, int line, bool abort = false);

#else
#  define DEVICE_ERROR_CHECK(ans)
#endif
#endif  // DEVICEERROR_H
