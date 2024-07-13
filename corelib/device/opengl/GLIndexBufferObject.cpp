#include "GLIndexBufferObject.h"
#include "DebugGL.h"

void GLIndexBufferObject::initialize() { GL_ERROR_CHECK(glGenBuffers(1, &id)); }
bool GLIndexBufferObject::isReady() const { return id != 0; }
void GLIndexBufferObject::bind() { GL_ERROR_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id)); }
void GLIndexBufferObject::unbind() { GL_ERROR_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)); }
void GLIndexBufferObject::clean() { GL_ERROR_CHECK(glDeleteBuffers(1, &id)); }
void GLIndexBufferObject::fill(const unsigned *buffer, size_t number_elements, DRAW_MODE mode) {
  GL_ERROR_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_elements, static_cast<const void *>(buffer), static_cast<GLenum>(mode)));
}