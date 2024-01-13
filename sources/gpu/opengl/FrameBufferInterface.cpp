#include "FrameBufferInterface.h"

FrameBufferInterface::FrameBufferInterface() {
  gl_framebuffer_object = nullptr;
  texture_dim = nullptr;
  texture_database = nullptr;
  default_framebuffer_pointer = nullptr;
}

FrameBufferInterface::FrameBufferInterface(TextureDatabase *_texture_database, Dim2 *_texture_dim, unsigned int *default_fbo)
    : FrameBufferInterface() {
  texture_dim = _texture_dim;
  texture_database = _texture_database;
  default_framebuffer_pointer = default_fbo;
  assert(texture_dim != nullptr);
}

FrameBufferInterface::~FrameBufferInterface() { gl_framebuffer_object = nullptr; }

FrameBufferInterface::FrameBufferInterface(FrameBufferInterface &&move) noexcept {
  std::swap(gl_framebuffer_object, move.gl_framebuffer_object);
  texture_dim = move.texture_dim;
  texture_database = move.texture_database;
  fbo_attachment_texture_collection = move.fbo_attachment_texture_collection;
  default_framebuffer_pointer = move.default_framebuffer_pointer;
  move.gl_framebuffer_object = nullptr;
  move.texture_dim = nullptr;
  move.texture_database = nullptr;
  move.fbo_attachment_texture_collection.clear();
  move.default_framebuffer_pointer = nullptr;
}

void FrameBufferInterface::resize() {
  if (texture_dim && gl_framebuffer_object) {
    for (auto A : fbo_attachment_texture_collection) {
      A.second->setNewSize(texture_dim->width, texture_dim->height);
      gl_framebuffer_object->resize(texture_dim->width, texture_dim->height);
    }
  }
}

void FrameBufferInterface::bindFrameBuffer() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->bind();
}

void FrameBufferInterface::unbindFrameBuffer() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->unbind();
}

void FrameBufferInterface::clean() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->clean();
}

void FrameBufferInterface::initializeFrameBuffer() {
  gl_framebuffer_object = std::make_unique<GLFrameBuffer>(
      texture_dim->width, texture_dim->height, GLRenderBuffer::DEPTH24_STENCIL8, default_framebuffer_pointer);
  gl_framebuffer_object->initializeBuffers();
}