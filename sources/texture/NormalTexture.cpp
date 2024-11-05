#include "NormalTexture.h"

#include "Shader.h"

void NormalTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + NORMAL);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (data)
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(NORMAL), NORMAL);
}

void NormalTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + NORMAL);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void NormalTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + NORMAL);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *NormalTexture::getTextureTypeCStr() { return type2str(NORMAL); }
