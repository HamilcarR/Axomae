#ifndef API_TEXTURE_H
#define API_TEXTURE_H
#include "api_common.h"
#include <cstdint>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <memory>
namespace nova {
  class Texture {
   public:
    virtual ~Texture() = default;
    virtual ERROR_STATE setData(const uint32_t *buffer) = 0;
    virtual ERROR_STATE setData(const float *buffer) = 0;
    virtual ERROR_STATE setWidth(unsigned w) = 0;
    virtual ERROR_STATE setHeight(unsigned h) = 0;
    virtual ERROR_STATE setChannels(unsigned c) = 0;
    virtual ERROR_STATE setInteropID(GLuint texture_id) = 0;
    virtual ERROR_STATE invertY() = 0;
    virtual ERROR_STATE invertX() = 0;
  };

  inline std::unique_ptr<Texture> create_texture();
}  // namespace nova

#endif
