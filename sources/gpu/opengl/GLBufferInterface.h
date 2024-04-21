#ifndef GLBufferInterface_H
#define GLBufferInterface_H

/**
 * @file GLBufferInterface.h
 * Defines an interface for Opengl buffer wrappers
 */

class GLBaseBufferInterface {
 public:
  virtual ~GLBaseBufferInterface() = default;
  virtual void initializeBuffers() = 0;
  [[nodiscard]] virtual bool isReady() const = 0;
  virtual void bind() = 0;
  virtual void unbind() = 0;
  virtual void clean() = 0;
};

class GLMutableBufferInterface : public GLBaseBufferInterface {
 public:
  virtual void fillBuffers() = 0;
};

class GLImmutableBufferInterface : public GLBaseBufferInterface {
 public:
  virtual void fillStorageBuffers(const void *data) = 0;
};
#endif