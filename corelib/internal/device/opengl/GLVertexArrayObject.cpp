#include "GLVertexArrayObject.h"
#include "init_3D.h"

void GLVertexArrayObject::initialize() { ax_glGenVertexArrays(1, &id); }

bool GLVertexArrayObject::isReady() const { return id != 0; }

void GLVertexArrayObject::bind() { ax_glBindVertexArray(id); }

void GLVertexArrayObject::unbind() { ax_glBindVertexArray(0); }

void GLVertexArrayObject::clean() {
  if (id != 0)
    ax_glDeleteVertexArrays(1, &id);
}