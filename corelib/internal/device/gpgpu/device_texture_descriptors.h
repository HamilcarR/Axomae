#ifndef DEVICE_TEXTURE_DESCRIPTORS_H
#define DEVICE_TEXTURE_DESCRIPTORS_H

#include "device_resource_descriptors.h"

namespace device::gpgpu {

  enum ADDRESS_MODE { ADDRESS_WRAP = 0x00, ADDRESS_CLAMP = 0x01, ADDRESS_MIRROR = 0x02, ADDRESS_BORDER = 0x03 };

  /**
   * FILTER_POINT = No interpolation when fetching texel
   * FILTER_LINEAR = Performs linear interpolation
   */
  enum FILTER_MODE { FILTER_POINT = 0x00, FILTER_LINEAR = 0x01 };

  /**
   * READ_ELEMENT_TYPE = fetch texel in whatever type it is stored
   * READ_NORMALIZED_FLOAT = divides texel value by it's max type to be in 0.f,1.f range
   */
  enum READ_MODE { READ_ELEMENT_TYPE = 0x00, READ_NORMALIZED_FLOAT = 0x01 };
  struct texture_descriptor {
    channel_format channel_descriptor;
    ADDRESS_MODE address_mode[3];
    READ_MODE read_mode;
    FILTER_MODE filter_mode;
    bool normalized_coords;
  };

  struct GPU_texture {
    std::any texture_object;
    std::any array_object;
    channel_format descriptor;
  };
}  // namespace device::gpgpu

#endif  // DEVICE_TEXTURE_DESCRIPTORS_H
