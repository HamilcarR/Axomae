#include "GLRenderBuffer.h"
#include "DebugGL.h"
GLRenderBuffer::GLRenderBuffer() {
  width = 0;
  height = 0;
  format = EMPTY;
}

GLRenderBuffer::GLRenderBuffer(unsigned int _width, unsigned int _height, INTERNAL_FORMAT _format) {
  width = _width;
  height = _height;
  format = _format;
}

void GLRenderBuffer::initializeBuffers() {
  GL_ERROR_CHECK(glGenRenderbuffers(1, &renderbuffer_id));
  bind();
  GL_ERROR_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, format, width, height));
}

bool GLRenderBuffer::isReady() const { return renderbuffer_id != 0; }

void GLRenderBuffer::fillBuffers() { EMPTY_FUNCBODY; }

void GLRenderBuffer::bind() { GL_ERROR_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer_id)); }

void GLRenderBuffer::unbind() { GL_ERROR_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, 0)); }

void GLRenderBuffer::clean() {
  unbind();
  GL_ERROR_CHECK(glDeleteRenderbuffers(1, &renderbuffer_id));
}

void GLRenderBuffer::resize(unsigned _width, unsigned _height) {
  bind();
  GL_ERROR_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, format, _width, _height));
  unbind();
  width = _width;
  height = _height;
}