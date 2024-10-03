#include "PackedGLGeometryBuffer.h"
#include "internal/device/rendering/opengl/DebugGL.h"

PackedGLGeometryBuffer::PackedGLGeometryBuffer() {
  geometry = nullptr;
  buffers_filled = false;
}

PackedGLGeometryBuffer::PackedGLGeometryBuffer(const Object3D *geometry) : PackedGLGeometryBuffer() { this->geometry = geometry; }

bool PackedGLGeometryBuffer::isReady() const {
  return vao.isReady() && vertex_buffer.isReady() && normal_buffer.isReady() && index_buffer.isReady() && texture_buffer.isReady() &&
         color_buffer.isReady() && tangent_buffer.isReady() && geometry;
}

void PackedGLGeometryBuffer::clean() {
  vao.unbind();
  vertex_buffer.unbind();
  normal_buffer.unbind();
  index_buffer.unbind();
  texture_buffer.unbind();
  color_buffer.unbind();
  tangent_buffer.unbind();

  vertex_buffer.clean();
  normal_buffer.clean();
  index_buffer.clean();
  texture_buffer.clean();
  color_buffer.clean();
  tangent_buffer.clean();
  vao.clean();
}

void PackedGLGeometryBuffer::initialize() {
  vao.initialize();
  vao.bind();
  vertex_buffer.initialize();
  normal_buffer.initialize();
  index_buffer.initialize();
  texture_buffer.initialize();
  color_buffer.initialize();
  tangent_buffer.initialize();
}

void PackedGLGeometryBuffer::bind() { bindVao(); }

void PackedGLGeometryBuffer::unbind() { unbindVao(); }

void PackedGLGeometryBuffer::bindVao() { vao.bind(); }

void PackedGLGeometryBuffer::unbindVao() { vao.unbind(); }

void PackedGLGeometryBuffer::bindVertexBuffer() { vertex_buffer.bind(); }

void PackedGLGeometryBuffer::bindNormalBuffer() { normal_buffer.bind(); }

void PackedGLGeometryBuffer::bindTextureBuffer() { texture_buffer.bind(); }

void PackedGLGeometryBuffer::bindColorBuffer() { color_buffer.bind(); }

void PackedGLGeometryBuffer::bindIndexBuffer() { index_buffer.bind(); }

void PackedGLGeometryBuffer::bindTangentBuffer() { tangent_buffer.bind(); }

// TODO: [AX-20] Provide methods to fill individual buffer , or to modify them
void PackedGLGeometryBuffer::fill() {
  if (!buffers_filled) {
    bindVao();
    bindVertexBuffer();
    vertex_buffer.fill(geometry->vertices.data(), geometry->vertices.size() * sizeof(float), GLVertexBufferObject<float>::STATIC);
    bindColorBuffer();
    color_buffer.fill(geometry->colors.data(), geometry->colors.size() * sizeof(float), GLVertexBufferObject<float>::STATIC);
    bindNormalBuffer();
    normal_buffer.fill(geometry->normals.data(), geometry->normals.size() * sizeof(float), GLVertexBufferObject<float>::STATIC);
    bindTextureBuffer();
    texture_buffer.fill(geometry->uv.data(), geometry->uv.size() * sizeof(float), GLVertexBufferObject<float>::STATIC);
    bindTangentBuffer();
    tangent_buffer.fill(geometry->tangents.data(), geometry->tangents.size() * sizeof(float), GLVertexBufferObject<float>::STATIC);
    bindIndexBuffer();
    index_buffer.fill(geometry->indices.data(), geometry->indices.size() * sizeof(unsigned), GLIndexBufferObject::STATIC);
    buffers_filled = true;
  }
}
