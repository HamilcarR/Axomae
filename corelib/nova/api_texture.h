#ifndef API_TEXTURE_H
#define API_TEXTURE_H
#include "api_common.h"
#include <cstdint>
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova {

  class Texture {
   public:
    virtual ~Texture() = default;

    template<class T>
    ERROR_STATE setTextureBuffer(const T *data) {
      return setData(data);
    }

    virtual ERROR_STATE setWidth(unsigned w) = 0;
    virtual ERROR_STATE setHeight(unsigned h) = 0;
    virtual ERROR_STATE setChannels(unsigned c) = 0;
    virtual ERROR_STATE setInteropID(GLuint texture_id) = 0;
    virtual ERROR_STATE invertY() = 0;
    virtual ERROR_STATE invertX() = 0;

    virtual const void *getTextureBuffer() const = 0;
    virtual unsigned getWidth() const = 0;
    virtual unsigned getHeight() const = 0;
    virtual unsigned getChannels() const = 0;
    virtual GLuint getInteropID() const = 0;
    virtual bool getInvertY() const = 0;
    virtual bool getInvertX() const = 0;
    virtual texture::FORMAT getFormat() const = 0;

   protected:
    virtual ERROR_STATE setData(const uint32_t *buffer) = 0;
    virtual ERROR_STATE setData(const float *buffer) = 0;
  };

  TexturePtr create_texture();
}  // namespace nova

#endif
