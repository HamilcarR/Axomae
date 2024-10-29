#include "FrameBufferTexture.h"
#include "Shader.h"

FrameBufferTexture::FrameBufferTexture() : GenericTexture() {
  internal_format = RGBA16F;
  data_format = BGRA;
  data_type = UBYTE;
}

FrameBufferTexture::FrameBufferTexture(unsigned _width, unsigned _height) : FrameBufferTexture() {
  width = _width;
  height = _height;
}

void FrameBufferTexture::initializeTexture2D() {
  ax_glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, nullptr);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void FrameBufferTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  FrameBufferTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(FRAMEBUFFER), FRAMEBUFFER);
}

void FrameBufferTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void FrameBufferTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *FrameBufferTexture::getTextureTypeCStr() { return type2str(FRAMEBUFFER); }
