#include "OpacityTexture.h"

#include "Shader.h"

void OpacityTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + OPACITY);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (data)
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(OPACITY), OPACITY);
}

void OpacityTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + OPACITY);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void OpacityTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + OPACITY);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *OpacityTexture::getTextureTypeCStr() { return type2str(OPACITY); }
