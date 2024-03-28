#include "GLFrameBuffer.h"
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
  assert(texture_id != 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, color_attachment, target, texture_id, mip_level);
}

void GLFrameBuffer::initializeBuffers() {
  glGenFramebuffers(1, &framebuffer_id);
  bind();
  if (renderbuffer_object != nullptr) {
    renderbuffer_object->initializeBuffers();
    if (renderbuffer_object->isReady()) {
      renderbuffer_object->bind();
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, DEPTH_STENCIL, GL_RENDERBUFFER, renderbuffer_object->getID());
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

void GLFrameBuffer::fillBuffers() { EMPTY_FUNCBODY; }

void GLFrameBuffer::bind() { glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_id); }

void GLFrameBuffer::unbind() {
  if (pointer_on_default_fbo_id)
    glBindFramebuffer(GL_FRAMEBUFFER, *pointer_on_default_fbo_id);
  else
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLFrameBuffer::clean() {
  if (renderbuffer_object != nullptr) {
    renderbuffer_object->clean();
  }
  unbind();
  glDeleteFramebuffers(1, &framebuffer_id);
}

void GLFrameBuffer::resize(unsigned _width, unsigned _height) {
  if (renderbuffer_object)
    renderbuffer_object->resize(_width, _height);
}