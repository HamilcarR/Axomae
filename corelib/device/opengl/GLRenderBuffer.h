#ifndef GLRENDERBUFFER_H
#define GLRENDERBUFFER_H
#include "GLBufferInterface.h"
#include "init_3D.h"
/**
 * @file GLRenderBuffer.h
 * Wrapper for OpenGL render buffers functions
 */

/**
 * @class GLRenderBuffer
 * @brief Provides a wrapper for render buffers
 */
class GLRenderBuffer : public GLMutableBufferInterface {
 public:
  enum INTERNAL_FORMAT : signed {
    EMPTY = -1,
    DEPTH16 = GL_DEPTH_COMPONENT16,
    DEPTH24 = GL_DEPTH_COMPONENT24,
    DEPTH32 = GL_DEPTH_COMPONENT32,
    DEPTH32F = GL_DEPTH_COMPONENT32F,
    DEPTH24_STENCIL8 = GL_DEPTH24_STENCIL8,
    DEPTH32F_STENCIL8 = GL_DEPTH32F_STENCIL8,
  };

 protected:
  unsigned int renderbuffer_id{};
  unsigned int width;
  unsigned int height;
  INTERNAL_FORMAT format;

 public:
  GLRenderBuffer();
  GLRenderBuffer(unsigned int width, unsigned int height, INTERNAL_FORMAT type);
  /**
   * @brief Creates a render buffer ID
   *
   */
  void initializeBuffers() override;
  [[nodiscard]] bool isReady() const override;
  void bind() override;
  void unbind() override;
  void clean() override;

  [[nodiscard]] unsigned int getID() const { return renderbuffer_id; }
  INTERNAL_FORMAT getFormat() { return format; }
  virtual void resize(unsigned width, unsigned height);
  void fillBuffers() override;
};

#endif