#include "SpecularTexture.h"
#include "Shader.h"

void SpecularTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + SPECULAR);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(SPECULAR), SPECULAR);
}

void SpecularTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + SPECULAR);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void SpecularTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + SPECULAR);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *SpecularTexture::getTextureTypeCStr() { return type2str(SPECULAR); }
