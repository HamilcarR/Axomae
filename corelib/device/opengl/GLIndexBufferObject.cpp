#include "GLIndexBufferObject.h"
#include "DebugGL.h"

void GLIndexBufferObject::initialize() { ax_glGenBuffers(1, &id); }
bool GLIndexBufferObject::isReady() const { return id != 0; }
void GLIndexBufferObject::bind() { ax_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id); }
void GLIndexBufferObject::unbind() { ax_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); }
void GLIndexBufferObject::clean() { ax_glDeleteBuffers(1, &id); }
void GLIndexBufferObject::fill(const unsigned *buffer, size_t number_elements, DRAW_MODE mode) {
  ax_glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_elements, static_cast<const void *>(buffer), static_cast<GLenum>(mode));
}