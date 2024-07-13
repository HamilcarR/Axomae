#include "FramebufferHelper.h"
#include "TextureDatabase.h"
FramebufferHelper::FramebufferHelper(TextureDatabase *_texture_database, Dim2 *_texture_dim, unsigned int *default_fbo) : FramebufferHelper() {
  texture_dim = _texture_dim;
  texture_database = _texture_database;
  default_framebuffer_pointer = default_fbo;
  AX_ASSERT_NOTNULL(texture_dim);
}

void FramebufferHelper::resize() {
  if (texture_dim && gl_framebuffer_object) {
    for (auto &A : fbo_attachment_texture_collection) {
      A.second->setNewSize(texture_dim->width, texture_dim->height);
      gl_framebuffer_object->resize(texture_dim->width, texture_dim->height);
    }
  }
}

void FramebufferHelper::bind() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->bind();
}

void FramebufferHelper::unbind() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->unbind();
}

void FramebufferHelper::clean() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->clean();
}
bool FramebufferHelper::isReady() const { return gl_framebuffer_object && texture_dim && default_framebuffer_pointer; }

void FramebufferHelper::initialize() {
  gl_framebuffer_object = std::make_unique<GLFrameBuffer>(
      texture_dim->width, texture_dim->height, GLRenderBuffer::DEPTH24_STENCIL8, default_framebuffer_pointer);
  gl_framebuffer_object->initialize();
}

Texture *FramebufferHelper::getFrameBufferTexturePointer(GLFrameBuffer::INTERNAL_FORMAT color_attachment) {
  return fbo_attachment_texture_collection[color_attachment];
}

void FramebufferHelper::setDefaultFrameBufferIdPointer(unsigned *id) { default_framebuffer_pointer = id; }

void FramebufferHelper::setTextureDimensions(Dim2 *pointer_on_texture_size) { texture_dim = pointer_on_texture_size; }
