#include "GLVertexArrayObject.h"
#include <DebugGL.h>

void GLVertexArrayObject::initialize() { GL_ERROR_CHECK(glGenVertexArrays(1, &id)); }

bool GLVertexArrayObject::isReady() const { return id != 0; }

void GLVertexArrayObject::bind() { GL_ERROR_CHECK(glBindVertexArray(id)); }

void GLVertexArrayObject::unbind() { GL_ERROR_CHECK(glBindVertexArray(0)); }

void GLVertexArrayObject::clean() {
  if (id != 0)
    GL_ERROR_CHECK(glDeleteVertexArrays(1, &id));
}