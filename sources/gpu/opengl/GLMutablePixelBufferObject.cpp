#include "GLPixelBufferObject.h"

#include "Axomae_macros.h"
#include "DebugGL.h"

GLPixelBufferObject::GLPixelBufferObject(TRANSFER t, size_t size) : buffer_type(t), buffer_size(size) {}

void GLPixelBufferObject::initializeBuffers() { GL_ERROR_CHECK(glGenBuffers(1, &pbo)); }

bool GLPixelBufferObject::isReady() const { return pbo != 0; }

void GLPixelBufferObject::fillSubBuffers(void *ptr, size_t offset, size_t size) { GL_ERROR_CHECK(glBufferSubData(buffer_type, offset, size, ptr)); }

void GLPixelBufferObject::bind() { GL_ERROR_CHECK(glBindBuffer(buffer_type, pbo)); }

void GLPixelBufferObject::unbind() { GL_ERROR_CHECK(glBindBuffer(buffer_type, 0)); }

void GLPixelBufferObject::clean() {
  unbind();
  GL_ERROR_CHECK(glDeleteBuffers(1, &pbo));
  pbo = 0;
  buffer_filled = false;
}

bool GLPixelBufferObject::unmapBuffer() { return GL_ERROR_CHECK(glUnmapBuffer(buffer_type)); }

void GLPixelBufferObject::setNewSize(size_t new_size) {
  setBufferSize(new_size);
  fillBuffers();
}

void GLMutablePixelBufferObject::fillBuffers() {
  GL_ERROR_CHECK(glBufferData(buffer_type, buffer_size, nullptr, GL_STREAM_DRAW));
  buffer_filled = true;
}

void GLImmutablePixelBufferObject::FillStorageBuffer(const void *data) {
  GL_ERROR_CHECK(glBufferStorage(buffer_type, buffer_size, data, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT));
  buffer_filled = true;
}
