#ifndef FramebufferInterface_H
#define FramebufferInterface_H
#include "DeviceBufferInterface.h"
#include "math_utils.h"

class FramebufferHelperInterface : public DeviceBaseBufferInterface {
 public:
  /**
   * @brief Resizes the textures used by the framebuffer .
   * Will use the values stored inside the texture_dim pointer property
   */
  virtual void resize() = 0;
  virtual void setTextureDimensions(Dim2 *pointer_on_texture_size) = 0;
  virtual void setDefaultFrameBufferIdPointer(unsigned *id) = 0;
};

#endif  // FramebufferInterface_H
