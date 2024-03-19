#ifndef GLBufferInterface_H
#define GLBufferInterface_H

/**
 * @file GLBufferInterface.h
 * Defines an interface for Opengl buffer wrappers
 *
 */

/**
 * @class GLBufferInterface
 */
class GLBufferInterface {
 public:
  /**
   * @brief Generates the framebuffer's ID.
   * !Note : This method should be called after the framebuffer texture generation as it uses : glFramebufferTexture2D
   *
   */
  virtual void initializeBuffers() = 0;
  /**
   * @brief Checks if framebuffer is ready to use
   *
   */
  virtual bool isReady() const = 0;
  virtual void fillBuffers() = 0;
  virtual void bind() = 0;
  virtual void unbind() = 0;
  virtual void clean() = 0;
};

#endif