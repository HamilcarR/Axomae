#ifndef DEVICE_RESOURCE_DATA_H
#define DEVICE_RESOURCE_DATA_H

#include "DeviceError.h"

namespace device::gpgpu {

  template<class T>
  struct gpu_query_result_t {
    T *device_ptr{};
    std::size_t size{};  // Always in bytes
    DeviceError error_status{};
  };

  using GPU_query_result = gpu_query_result_t<void>;

}  // namespace device::gpgpu
#endif
