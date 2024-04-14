#ifndef GLPIXELBUFFEROBJECT_H
#define GLPIXELBUFFEROBJECT_H
#include "GLBufferInterface.h"
#include "init_3D.h"
class GLPixelBufferObject final : public GLBufferInterface {
 public:
  enum TRANSFER : unsigned { DOWN = GL_PIXEL_PACK_BUFFER, UP = GL_PIXEL_UNPACK_BUFFER };
  enum ACCESS : unsigned { R = GL_READ_ONLY, W = GL_WRITE_ONLY, RW = GL_READ_WRITE };

 private:
  GLuint pbo{0};
  TRANSFER buffer_type{UP};
  size_t buffer_size;
  bool buffer_filled{false};
  ACCESS access_type{R};

 public:
  GLPixelBufferObject() = default;
  GLPixelBufferObject(TRANSFER type, size_t size);
  ~GLPixelBufferObject() override = default;

  void initializeBuffers() override;
  [[nodiscard]] bool isReady() const override;
  void fillBuffers() override;
  void fillSubBuffers(void *buffer, size_t offset, size_t data_size);
  void bind() override;
  void unbind() override;
  void clean() override;
  template<class T>
  T *mapBuffer(ACCESS access);
  bool unmapBuffer();
  void setBufferSize(size_t new_size) { buffer_size = new_size; }
  void setNewSize(size_t new_size);
  void setTransferType(TRANSFER new_type) { buffer_type = new_type; }
  void setAccessType(ACCESS new_access_type) { access_type = new_access_type; }
};
template<class T>
T *GLPixelBufferObject::mapBuffer(ACCESS access) {
  return static_cast<T *>(glMapBuffer(buffer_type, access));
}
#endif
