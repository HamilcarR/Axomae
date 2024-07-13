#include "GLFrameBuffer.h"
#include "DebugGL.h"
#include "Logger.h"

GLFrameBuffer::GLFrameBuffer() {
  renderbuffer_object = nullptr;
  framebuffer_id = 0;
  texture_id = 0;
}

GLFrameBuffer::GLFrameBuffer(
    unsigned _width, unsigned _height, GLRenderBuffer::INTERNAL_FORMAT rbo_format, unsigned int *default_fbo_id_pointer, TEXTURE_TARGET target_type)
    : GLFrameBuffer() {
  if (rbo_format != GLRenderBuffer::EMPTY)
    renderbuffer_object = std::make_unique<GLRenderBuffer>(_width, _height, rbo_format);
  target_texture_type = target_type;
  pointer_on_default_fbo_id = default_fbo_id_pointer;
}

void GLFrameBuffer::attachTexture2D(INTERNAL_FORMAT color_attachment, TEXTURE_TARGET target, unsigned int texture_id, unsigned mip_level) {
  AX_ASSERT_NEQ(texture_id, 0);
  ax_glFramebufferTexture2D(GL_FRAMEBUFFER, color_attachment, target, texture_id, mip_level);
}

void GLFrameBuffer::initialize() {
  ax_glGenFramebuffers(1, &framebuffer_id);
  bind();
  if (renderbuffer_object != nullptr) {
    renderbuffer_object->initialize();
    if (renderbuffer_object->isReady()) {
      renderbuffer_object->bind();
      ax_glFramebufferRenderbuffer(GL_FRAMEBUFFER, DEPTH_STENCIL, GL_RENDERBUFFER, renderbuffer_object->getID());
    } else
      LOG("Problem initializing render buffer", LogLevel::ERROR);
    auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      LOG("Framebuffer not ready to use , server returned : " + std::to_string(status), LogLevel::ERROR);
    }
  }
  unbind();
}

bool GLFrameBuffer::isReady() const { return framebuffer_id != 0; }

void GLFrameBuffer::fill() { AX_UNREACHABLE }

void GLFrameBuffer::bind() { ax_glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_id); }

void GLFrameBuffer::unbind() {
  if (pointer_on_default_fbo_id) {
    ax_glBindFramebuffer(GL_FRAMEBUFFER, *pointer_on_default_fbo_id);
  } else {
    ax_glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
}

void GLFrameBuffer::clean() {
  if (renderbuffer_object != nullptr) {
    renderbuffer_object->clean();
  }
  unbind();
  ax_glDeleteFramebuffers(1, &framebuffer_id);
}

void GLFrameBuffer::resize(unsigned _width, unsigned _height) {
  if (renderbuffer_object)
    renderbuffer_object->resize(_width, _height);
}