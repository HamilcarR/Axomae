//
// Created by hamilcar on 7/13/24.
//

#include "MetallicTexture.h"

#include "Shader.h"
MetallicTexture::MetallicTexture() { type = METALLIC; }

MetallicTexture::MetallicTexture(TextureData *data) : GenericTexture(data) {
  type = METALLIC;
  if (!data)
    set_dummy_TextureData(this);
}

void MetallicTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + METALLIC);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(METALLIC), METALLIC);
}

void MetallicTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + METALLIC);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void MetallicTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + METALLIC);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *MetallicTexture::getTextureTypeCStr() { return type2str(METALLIC); }