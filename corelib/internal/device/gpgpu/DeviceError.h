#ifndef DEVICEERROR_H
#define DEVICEERROR_H
#include "internal/macro/project_macros.h"
#include <cstdint>
#include <driver_types.h>

class DeviceError {

#if defined(AXOMAE_USE_CUDA)
 private:
  cudaError_t id;

 public:
  explicit DeviceError(cudaError_t error);
  [[nodiscard]] cudaError_t getId() const { return id; }
#endif
 public:
  CLASS_CM(DeviceError)
  [[nodiscard]] bool isOk() const;
};

#if defined(AXOMAE_USE_CUDA)

#  define DEVICE_ERROR_CHECK(ans) \
    { \
      gpgpu_err_log((ans), __FILE__, __LINE__); \
    }
void gpgpu_err_log(const DeviceError &err, const char *file, int line, bool abort = false);

#else
#  define DEVICE_ERROR_CHECK(ans)
#endif
#endif  // DEVICEERROR_H
