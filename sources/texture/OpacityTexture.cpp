//
// Created by hamilcar on 7/13/24.
//

#include "OpacityTexture.h"

#include "Shader.h"
OpacityTexture::OpacityTexture() { type = OPACITY; }

OpacityTexture::OpacityTexture(TextureData *data) : GenericTexture(data) {
  type = OPACITY;
  if (!data)
    set_dummy_TextureData(this);
}

void OpacityTexture::setGlData(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + OPACITY);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(OPACITY), OPACITY);
}

void OpacityTexture::bindTexture() {
  ax_glActiveTexture(GL_TEXTURE0 + OPACITY);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void OpacityTexture::unbindTexture() {
  ax_glActiveTexture(GL_TEXTURE0 + OPACITY);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *OpacityTexture::getTextureTypeCStr() { return type2str(OPACITY); }
