//
// Created by hamilcar on 7/13/24.
//

#include "SpecularTexture.h"
#include "Shader.h"
SpecularTexture::SpecularTexture() { type = SPECULAR; }

SpecularTexture::SpecularTexture(TextureData *data) : GenericTexture(data) {
  type = SPECULAR;
  if (!data)
    set_dummy_TextureData(this);
}

void SpecularTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + SPECULAR);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(SPECULAR), SPECULAR);
}

void SpecularTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + SPECULAR);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void SpecularTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + SPECULAR);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *SpecularTexture::getTextureTypeCStr() { return type2str(SPECULAR); }
