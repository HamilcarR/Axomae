#ifndef GLVERTEXBUFFEROBJECT_H
#define GLVERTEXBUFFEROBJECT_H
#include "../DeviceBufferInterface.h"
#include "DebugGL.h"
#include "init_3D.h"
#include "project_macros.h"

template<class T>
class GLVertexBufferObject final : public DeviceBaseBufferInterface {
 public:
  enum DRAW_MODE : int { STATIC = GL_STATIC_DRAW, DYNAMIC = GL_DYNAMIC_DRAW, STREAM = GL_STREAM_DRAW };

 private:
  GLuint id{0};

 public:
  CLASS_OCM(GLVertexBufferObject)

  void initialize() override;
  [[nodiscard]] bool isReady() const override;
  void bind() override;
  void unbind() override;
  void clean() override;
  void fill(const T *buffer, size_t elements, DRAW_MODE draw_mode);
};

template<class T>
void GLVertexBufferObject<T>::initialize() {
  GL_ERROR_CHECK(glGenBuffers(1, &id));
}

template<class T>
bool GLVertexBufferObject<T>::isReady() const {
  return id != 0;
}

template<class T>
void GLVertexBufferObject<T>::bind() {
  GL_ERROR_CHECK(glBindBuffer(GL_ARRAY_BUFFER, id));
}

template<class T>
void GLVertexBufferObject<T>::unbind() {
  GL_ERROR_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

template<class T>
void GLVertexBufferObject<T>::clean() {
  if (id != 0)
    GL_ERROR_CHECK(glDeleteBuffers(1, &id));
}

template<class T>
void __attribute((optimize("O0"))) GLVertexBufferObject<T>::fill(const T *buffer, size_t elements, DRAW_MODE draw) {
  GL_ERROR_CHECK(glBufferData(GL_ARRAY_BUFFER, elements, static_cast<const void *>(buffer), static_cast<GLenum>(draw)));
}
#endif  // GLVERTEXBUFFEROBJECT_H
