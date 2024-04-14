#include "GLGeometryBuffer.h"
#include "DebugGL.h"
GLGeometryBuffer::GLGeometryBuffer() {
  geometry = nullptr;
  vao = 0;
  vertex_buffer = 0;
  normal_buffer = 0;
  index_buffer = 0;
  texture_buffer = 0;
  color_buffer = 0;
  tangent_buffer = 0;
  buffers_filled = false;
}

GLGeometryBuffer::GLGeometryBuffer(const Object3D *geometry) : GLGeometryBuffer() { this->geometry = geometry; }

bool GLGeometryBuffer::isReady() const {
  return vao && vertex_buffer && normal_buffer && index_buffer && texture_buffer && color_buffer && tangent_buffer && geometry;
}

void GLGeometryBuffer::clean() {
  unbindVao();
  if (vertex_buffer != 0) {
    GL_ERROR_CHECK(glDeleteBuffers(1, &vertex_buffer));
  }
  if (normal_buffer != 0) {
    GL_ERROR_CHECK(glDeleteBuffers(1, &normal_buffer));
  }
  if (index_buffer != 0) {
    GL_ERROR_CHECK(glDeleteBuffers(1, &index_buffer));
  }
  if (texture_buffer != 0) {
    GL_ERROR_CHECK(glDeleteBuffers(1, &texture_buffer));
  }
  if (color_buffer != 0) {
    GL_ERROR_CHECK(glDeleteBuffers(1, &color_buffer));
  }
  if (tangent_buffer != 0) {
    GL_ERROR_CHECK(glDeleteBuffers(1, &tangent_buffer));
  }
  if (vao != 0) {
    GL_ERROR_CHECK(glDeleteVertexArrays(1, &vao));
  }
}

void GLGeometryBuffer::initializeBuffers() {
  GL_ERROR_CHECK(glGenVertexArrays(1, &vao));
  bindVao();
  GL_ERROR_CHECK(glGenBuffers(1, &vertex_buffer));
  GL_ERROR_CHECK(glGenBuffers(1, &normal_buffer));
  GL_ERROR_CHECK(glGenBuffers(1, &color_buffer));
  GL_ERROR_CHECK(glGenBuffers(1, &texture_buffer));
  GL_ERROR_CHECK(glGenBuffers(1, &tangent_buffer));
  GL_ERROR_CHECK(glGenBuffers(1, &index_buffer));
}

void GLGeometryBuffer::bind() { bindVao(); }

void GLGeometryBuffer::unbind() { unbindVao(); }

void GLGeometryBuffer::bindVao() { GL_ERROR_CHECK(glBindVertexArray(vao)); }

void GLGeometryBuffer::bindVertexBuffer() { GL_ERROR_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)); }

void GLGeometryBuffer::bindNormalBuffer() { GL_ERROR_CHECK(glBindBuffer(GL_ARRAY_BUFFER, normal_buffer)); }

void GLGeometryBuffer::bindTextureBuffer() { GL_ERROR_CHECK(glBindBuffer(GL_ARRAY_BUFFER, texture_buffer)); }

void GLGeometryBuffer::bindColorBuffer() { GL_ERROR_CHECK(glBindBuffer(GL_ARRAY_BUFFER, color_buffer)); }

void GLGeometryBuffer::bindIndexBuffer() { GL_ERROR_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)); }

void GLGeometryBuffer::bindTangentBuffer() { GL_ERROR_CHECK(glBindBuffer(GL_ARRAY_BUFFER, tangent_buffer)); }

void GLGeometryBuffer::unbindVao() { GL_ERROR_CHECK(glBindVertexArray(0)); }

// TODO: [AX-20] Provide methods to fill individual buffer , or to modify them
void GLGeometryBuffer::fillBuffers() {
  if (!buffers_filled) {
    bindVao();
    bindVertexBuffer();
    GL_ERROR_CHECK(glBufferData(GL_ARRAY_BUFFER, geometry->vertices.size() * sizeof(float), geometry->vertices.data(), GL_STATIC_DRAW));
    bindColorBuffer();
    GL_ERROR_CHECK(glBufferData(GL_ARRAY_BUFFER, geometry->colors.size() * sizeof(float), geometry->colors.data(), GL_STATIC_DRAW));
    bindNormalBuffer();
    GL_ERROR_CHECK(glBufferData(GL_ARRAY_BUFFER, geometry->normals.size() * sizeof(float), geometry->normals.data(), GL_STATIC_DRAW));
    bindTextureBuffer();
    GL_ERROR_CHECK(glBufferData(GL_ARRAY_BUFFER, geometry->uv.size() * sizeof(float), geometry->uv.data(), GL_STATIC_DRAW));
    bindTangentBuffer();
    GL_ERROR_CHECK(glBufferData(GL_ARRAY_BUFFER, geometry->tangents.size() * sizeof(float), geometry->tangents.data(), GL_STATIC_DRAW));
    bindIndexBuffer();
    GL_ERROR_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, geometry->indices.size() * sizeof(unsigned int), geometry->indices.data(), GL_STATIC_DRAW));
    buffers_filled = true;
  }
}
