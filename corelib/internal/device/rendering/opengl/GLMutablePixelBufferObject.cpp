#include "GLMutablePixelBufferObject.h"

#include "DebugGL.h"
#include "internal/macro/project_macros.h"

GLMutablePixelBufferObject::GLMutablePixelBufferObject(TRANSFER t, size_t size) : buffer_type(t), buffer_size(size) {}

void GLMutablePixelBufferObject::initialize() { ax_glGenBuffers(1, &pbo); }

bool GLMutablePixelBufferObject::isReady() const { return pbo != 0; }

void GLMutablePixelBufferObject::fillSubBuffers(void *ptr, size_t offset, size_t size) { ax_glBufferSubData(buffer_type, offset, size, ptr); }

void GLMutablePixelBufferObject::bind() { ax_glBindBuffer(buffer_type, pbo); }

void GLMutablePixelBufferObject::unbind() { ax_glBindBuffer(buffer_type, 0); }

void GLMutablePixelBufferObject::clean() {
  unbind();
  ax_glDeleteBuffers(1, &pbo);
  pbo = 0;
  buffer_filled = false;
}

bool GLMutablePixelBufferObject::unmapBuffer() {
  bool ret = ax_glUnmapBuffer(buffer_type);
  return ret;
}

void GLMutablePixelBufferObject::setNewSize(size_t new_size) {
  setBufferSize(new_size);
  fill();
}

void GLMutablePixelBufferObject::fill() {
  ax_glBufferData(buffer_type, buffer_size, nullptr, GL_DYNAMIC_DRAW);
  buffer_filled = true;
}
void GLMutablePixelBufferObject::fillBuffersAddress(void *ptr) {
  ax_glBufferData(buffer_type, buffer_size, ptr, GL_DYNAMIC_DRAW);
  buffer_filled = true;
}

void GLMutablePixelBufferObject::flushMappedRange(size_t offset, size_t length) { ax_glFlushMappedBufferRange(buffer_type, offset, length); }
