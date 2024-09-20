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

void GLRenderBuffer::initialize() {
  ax_glGenRenderbuffers(1, &renderbuffer_id);
  bind();
  ax_glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
}

bool GLRenderBuffer::isReady() const { return renderbuffer_id != 0; }

void GLRenderBuffer::fill() { EMPTY_FUNCBODY }

void GLRenderBuffer::bind() { ax_glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer_id); }

void GLRenderBuffer::unbind() { ax_glBindRenderbuffer(GL_RENDERBUFFER, 0); }

void GLRenderBuffer::clean() {
  unbind();
  ax_glDeleteRenderbuffers(1, &renderbuffer_id);
}

void GLRenderBuffer::resize(unsigned _width, unsigned _height) {
  bind();
  ax_glRenderbufferStorage(GL_RENDERBUFFER, format, _width, _height);
  unbind();
  width = _width;
  height = _height;
}