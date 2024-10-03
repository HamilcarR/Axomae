#ifndef GLPIXELBUFFEROBJECT_H
#define GLPIXELBUFFEROBJECT_H
#include "../DeviceBufferInterface.h"
#include "DebugGL.h"
#include "init_3D.h"
#include "internal/macro/project_macros.h"

class GLMutablePixelBufferObject final : public DeviceMutableBufferInterface {
 public:
  enum TRANSFER : unsigned { DOWN = GL_PIXEL_PACK_BUFFER, UP = GL_PIXEL_UNPACK_BUFFER };
  enum ACCESS : unsigned { R = GL_READ_ONLY, W = GL_WRITE_ONLY, RW = GL_READ_WRITE };

 protected:
  GLuint pbo{0};
  TRANSFER buffer_type{UP};
  size_t buffer_size{0};
  bool buffer_filled{false};
  ACCESS access_type{R};

 public:
  CLASS_OCM(GLMutablePixelBufferObject)

  GLMutablePixelBufferObject(TRANSFER type, size_t size);

  void initialize() override;
  [[nodiscard]] bool isReady() const override;
  void fill() override;
  void fillBuffersAddress(void *address);
  void fillSubBuffers(void *buffer, size_t offset, size_t length);
  void flushMappedRange(size_t offset, size_t length);
  void bind() override;
  void unbind() override;
  void clean() override;
  template<class T>
  T *mapBuffer(ACCESS access);
  template<class T>
  T *mapBufferRange(size_t offset, size_t size, GLbitfield flag);
  bool unmapBuffer();
  void setBufferSize(size_t new_size) { buffer_size = new_size; }
  void setNewSize(size_t new_size);
  void setTransferType(TRANSFER new_type) { buffer_type = new_type; }
  void setAccessType(ACCESS new_access_type) { access_type = new_access_type; }
  [[nodiscard]] TRANSFER getTransferType() const { return buffer_type; }
};
template<class T>
T *GLMutablePixelBufferObject::mapBuffer(ACCESS access) {
  return static_cast<T *>(ax_glMapBuffer(buffer_type, access));
}

template<class T>
T *GLMutablePixelBufferObject::mapBufferRange(size_t offset, size_t size, GLbitfield flag) {
  return static_cast<T *>(ax_glMapBufferRange(buffer_type, offset, size, GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | flag));
}

#endif
