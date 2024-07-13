#include "SpecularTexture.h"
#include "Shader.h"
SpecularTexture::SpecularTexture() { type = SPECULAR; }

SpecularTexture::SpecularTexture(TextureData *data) : GenericTexture(data) {
  type = SPECULAR;
  if (!data)
    set_dummy_TextureData(this);
}

void SpecularTexture::setGlData(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + SPECULAR);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(SPECULAR), SPECULAR);
}

void SpecularTexture::bindTexture() {
  ax_glActiveTexture(GL_TEXTURE0 + SPECULAR);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void SpecularTexture::unbindTexture() {
  ax_glActiveTexture(GL_TEXTURE0 + SPECULAR);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *SpecularTexture::getTextureTypeCStr() { return type2str(SPECULAR); }
