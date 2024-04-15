#ifndef FramebufferInterface_H
#define FramebufferInterface_H
#include "constants.h"

class Texture;

class FramebufferInterface {
 public:
  virtual ~FramebufferInterface() = default;
  /**
   * @brief Resizes the textures used by the framebuffer .
   * Will use the values stored inside the texture_dim pointer property
   */
  virtual void resize() = 0;
  virtual void setTextureDimensions(Dim2 *pointer_on_texture_size) = 0;
  virtual void bindFrameBuffer() = 0;
  virtual void unbindFrameBuffer() = 0;
  virtual void initializeFrameBuffer() = 0;
  virtual void clean() = 0;
  virtual void setDefaultFrameBufferIdPointer(unsigned *id) = 0;
};

#endif  // FramebufferInterface_H
