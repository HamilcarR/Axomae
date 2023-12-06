#include "../includes/FrameBufferInterface.h"

FrameBufferInterface::FrameBufferInterface() {
  gl_framebuffer_object = nullptr;
  texture_dim = nullptr;
  texture_database = nullptr;
  default_framebuffer_pointer = nullptr;
}

FrameBufferInterface::FrameBufferInterface(TextureDatabase *_texture_database,
                                           ScreenSize *_texture_dim,
                                           unsigned int *default_fbo)
    : FrameBufferInterface() {
  texture_dim = _texture_dim;
  texture_database = _texture_database;
  default_framebuffer_pointer = default_fbo;
  assert(texture_dim != nullptr);
}

FrameBufferInterface::~FrameBufferInterface() {
  if (gl_framebuffer_object != nullptr)
    delete gl_framebuffer_object;
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
  gl_framebuffer_object = new GLFrameBuffer(
      texture_dim->width, texture_dim->height, GLRenderBuffer::DEPTH24_STENCIL8, default_framebuffer_pointer);
  gl_framebuffer_object->initializeBuffers();
}