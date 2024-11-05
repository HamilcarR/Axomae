#include "MetallicTexture.h"

#include "Shader.h"

void MetallicTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + METALLIC);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (data)
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(METALLIC), METALLIC);
}

void MetallicTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + METALLIC);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void MetallicTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + METALLIC);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *MetallicTexture::getTextureTypeCStr() { return type2str(METALLIC); }