#ifndef DEVICE_RESOURCE_DATA_H
#define DEVICE_RESOURCE_DATA_H

#include "DeviceError.h"

namespace device::gpgpu {

  struct GPU_query_result {
    void *device_ptr{};
    DeviceError error_status;
  };

}  // namespace device::gpgpu
#endif
