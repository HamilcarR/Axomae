#ifndef GLVERTEXBUFFEROBJECT_H
#define GLVERTEXBUFFEROBJECT_H
#include "../DeviceBufferInterface.h"
#include "DebugGL.h"
#include "init_3D.h"
#include "internal/macro/project_macros.h"

template<class T>
class GLVertexBufferObject final : public DeviceBaseBufferInterface {
 public:
  enum DRAW_MODE : int { STATIC = GL_STATIC_DRAW, DYNAMIC = GL_DYNAMIC_DRAW, STREAM = GL_STREAM_DRAW };

 private:
  GLuint id{0};

 public:
  CLASS_OCM(GLVertexBufferObject)

  void initialize() override;
  ax_no_discard bool isReady() const override;
  void bind() override;
  void unbind() override;
  void clean() override;
  void fill(const T *buffer, size_t elements, DRAW_MODE draw_mode);
};

template<class T>
void GLVertexBufferObject<T>::initialize() {
  ax_glGenBuffers(1, &id);
}

template<class T>
bool GLVertexBufferObject<T>::isReady() const {
  return id != 0;
}

template<class T>
void GLVertexBufferObject<T>::bind() {
  ax_glBindBuffer(GL_ARRAY_BUFFER, id);
}

template<class T>
void GLVertexBufferObject<T>::unbind() {
  ax_glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template<class T>
void GLVertexBufferObject<T>::clean() {
  if (id != 0)
    ax_glDeleteBuffers(1, &id);
}

template<class T>
void GLVertexBufferObject<T>::fill(const T *buffer, size_t elements, DRAW_MODE draw) {
  ax_glBufferData(GL_ARRAY_BUFFER, elements, static_cast<const void *>(buffer), static_cast<GLenum>(draw));
}
#endif  // GLVERTEXBUFFEROBJECT_H
