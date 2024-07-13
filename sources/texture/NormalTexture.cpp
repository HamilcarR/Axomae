//
// Created by hamilcar on 7/13/24.
//

#include "NormalTexture.h"

#include "Shader.h"
NormalTexture::NormalTexture() { type = NORMAL; }

NormalTexture::NormalTexture(TextureData *texture) : GenericTexture(texture) {
  type = NORMAL;
  if (data.empty())
    set_dummy_TextureData_normals(this);
}

void NormalTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + NORMAL);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(type), type);
}

void NormalTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + NORMAL);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void NormalTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + NORMAL);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *NormalTexture::getTextureTypeCStr() { return type2str(NORMAL); }
