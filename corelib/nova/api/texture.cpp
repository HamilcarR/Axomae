#include "../api.h"
#include "api_common.h"
#include "private_includes.h"
#include <memory>
namespace nova {
  NvTexture::NvTexture(const NvAbstractTexture &other) { *this = *dynamic_cast<const NvTexture *>(&other); }

  NvTexture &NvTexture::operator=(const NvAbstractTexture &other) {
    if (&other == this)
      return *this;
    *this = *dynamic_cast<const NvTexture *>(&other);
    return *this;
  }

  ERROR_STATE NvTexture::setData(const uint32_t *buffer) {
    if (!buffer)
      return INVALID_BUFFER_STATE;
    memory.ui_buffer = buffer;
    type = I_ARRAY;
    return SUCCESS;
  }

  ERROR_STATE NvTexture::setData(const float *buffer) {
    if (!buffer)
      return INVALID_BUFFER_STATE;
    memory.f_buffer = buffer;
    type = F_ARRAY;
    return SUCCESS;
  }

  ERROR_STATE NvTexture::setWidth(unsigned w) {
    width = w;
    return SUCCESS;
  }

  ERROR_STATE NvTexture::setHeight(unsigned h) {
    height = h;
    return SUCCESS;
  }

  ERROR_STATE NvTexture::setChannels(unsigned c) {
    if (c < 1 || c > 4)
      return INVALID_CHANNEL_DESCRIPTOR;
    channel = c;
    return SUCCESS;
  }

  ERROR_STATE NvTexture::setInteropID(GLuint texture_id) {
    interop_id = texture_id;
    return SUCCESS;
  }

  ERROR_STATE NvTexture::invertY() {
    invert_y ^= true;
    return SUCCESS;
  }

  ERROR_STATE NvTexture::invertX() {
    invert_x ^= true;
    return SUCCESS;
  }

  template<>
  const uint32_t *NvTexture::getData<uint32_t>() const {
    return memory.ui_buffer;
  }

  template<>
  const float *NvTexture::getData<float>() const {
    return memory.f_buffer;
  }

  std::unique_ptr<NvAbstractTexture> create_texture() { return std::make_unique<NvTexture>(); }
}  // namespace nova
