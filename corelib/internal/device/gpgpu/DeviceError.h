#ifndef DEVICEERROR_H
#define DEVICEERROR_H
#include "internal/macro/project_macros.h"
#include <cstdint>
#if defined(AXOMAE_USE_CUDA)
#  include <driver_types.h>
class DeviceError;
#  define DEVICE_ERROR_CHECK(ans) \
    { \
      gpgpu_err_log((ans), __FILE__, __LINE__); \
    }
void gpgpu_err_log(const DeviceError &err, const char *file, int line, bool abort = false);
void gpgpu_err_log(cudaError_t err, const char *file, int line, bool abort = false);

#else
#  define DEVICE_ERROR_CHECK(ans)
#endif

/* Replace with pimpl instead of conditional compilation*/
class DeviceError {

#if defined(AXOMAE_USE_CUDA)
 private:
  cudaError_t id;

 public:
  DeviceError(cudaError_t error);
  ax_no_discard cudaError_t getId() const { return id; }
#endif
 public:
  CLASS_CM(DeviceError)
  ax_no_discard bool isValid() const;
};

#endif  // DEVICEERROR_H
