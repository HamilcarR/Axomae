//
// Created by hamilcar on 7/13/24.
//

#include "GenericCubemapTexture.h"
#include "Shader.h"

GenericCubemapTexture::GenericCubemapTexture(FORMAT internal_format, FORMAT data_format, FORMAT data_type, unsigned width, unsigned height)
    : CubemapTexture(internal_format, data_format, data_type, width, height) {
  type = GENERIC_CUBE;
}

GenericCubemapTexture::GenericCubemapTexture(TextureData *data) : CubemapTexture(data) { type = GENERIC_CUBE; }

void GenericCubemapTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_CUBE_MAP, sampler2D);
}

void GenericCubemapTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void GenericCubemapTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  bindTexture();
  initializeTexture2D();
  shader->setTextureUniforms(location_name, texture_unit);
  unbindTexture();
}

const char *GenericCubemapTexture::getTextureTypeCStr() { return location_name.c_str(); }
