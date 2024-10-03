#ifndef DEVICE_RESOURCE_DATA_H
#define DEVICE_RESOURCE_DATA_H

#include "DeviceError.h"
#include <any>

namespace device::gpgpu::resource {

  struct GPU_resource {
    void *device_ptr{};
    DeviceError error_status;
  };

  enum TEXTURE_TYPE { FLOAT = 0 };

  struct texture_channel_descriptor {
    int bits_size_x, bits_size_y, bits_size_z, bits_size_a;
    TEXTURE_TYPE texture_type;
  };

  struct GPU_array {
    std::any array;
  };

  struct GPU_texture {
    std::any texture_object;
    std::any array_object;
    texture_channel_descriptor descriptor;
  };

}  // namespace device::gpgpu::resource
#endif
