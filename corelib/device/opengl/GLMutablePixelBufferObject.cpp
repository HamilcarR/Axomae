#include "GLMutablePixelBufferObject.h"

#include "DebugGL.h"
#include "project_macros.h"

GLMutablePixelBufferObject::GLMutablePixelBufferObject(TRANSFER t, size_t size) : buffer_type(t), buffer_size(size) {}

void GLMutablePixelBufferObject::initialize() { GL_ERROR_CHECK(glGenBuffers(1, &pbo)); }

bool GLMutablePixelBufferObject::isReady() const { return pbo != 0; }

void GLMutablePixelBufferObject::fillSubBuffers(void *ptr, size_t offset, size_t size) {
  GL_ERROR_CHECK(glBufferSubData(buffer_type, offset, size, ptr));
}

void GLMutablePixelBufferObject::bind() { GL_ERROR_CHECK(glBindBuffer(buffer_type, pbo)); }

void GLMutablePixelBufferObject::unbind() { GL_ERROR_CHECK(glBindBuffer(buffer_type, 0)); }

void GLMutablePixelBufferObject::clean() {
  unbind();
  GL_ERROR_CHECK(glDeleteBuffers(1, &pbo));
  pbo = 0;
  buffer_filled = false;
}

bool GLMutablePixelBufferObject::unmapBuffer() {
  bool ret = glUnmapBuffer(buffer_type);
  errorCheck(__FILE__, __func__, __LINE__);
  return ret;
}

void GLMutablePixelBufferObject::setNewSize(size_t new_size) {
  setBufferSize(new_size);
  fill();
}

void GLMutablePixelBufferObject::fill() {
  GL_ERROR_CHECK(glBufferData(buffer_type, buffer_size, nullptr, GL_DYNAMIC_DRAW));
  buffer_filled = true;
}
void GLMutablePixelBufferObject::fillBuffersAddress(void *ptr) {
  GL_ERROR_CHECK(glBufferData(buffer_type, buffer_size, ptr, GL_DYNAMIC_DRAW));
  buffer_filled = true;
}

void GLMutablePixelBufferObject::flushMappedRange(size_t offset, size_t length) { glFlushMappedBufferRange(buffer_type, offset, length); }

/*void GLImmutablePixelBufferObject::FillStorageBuffer(const void *data) {
  GL_ERROR_CHECK(glBufferStorage(buffer_type, buffer_size, data, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT));
  buffer_filled = true;
}*/
