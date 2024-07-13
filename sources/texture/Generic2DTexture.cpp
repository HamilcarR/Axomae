//
// Created by hamilcar on 7/13/24.
//

#include "Generic2DTexture.h"
#include "Shader.h"

Generic2DTexture::Generic2DTexture() { type = GENERIC; }

Generic2DTexture::Generic2DTexture(TextureData *data) : GenericTexture(data) { type = GENERIC; }

void Generic2DTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(location_name, texture_unit);
}

void Generic2DTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void Generic2DTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *Generic2DTexture::getTextureTypeCStr() { return location_name.c_str(); }

void Generic2DTexture::setTextureUnit(unsigned int tex_unit) { texture_unit = tex_unit; }

void Generic2DTexture::setLocationName(const std::string &_name) { location_name = _name; }
