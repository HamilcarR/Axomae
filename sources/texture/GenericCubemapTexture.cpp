#include "GenericCubemapTexture.h"
#include "Shader.h"

GenericCubemapTexture::GenericCubemapTexture(FORMAT internal_format, FORMAT data_format, FORMAT data_type, unsigned width, unsigned height)
    : CubemapTexture(internal_format, data_format, data_type, width, height) {}

GenericCubemapTexture::GenericCubemapTexture(TextureData *data) : CubemapTexture(data) {}

void GenericCubemapTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + texture_unit);
  ax_glBindTexture(GL_TEXTURE_CUBE_MAP, sampler2D);
}

void GenericCubemapTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + texture_unit);
  ax_glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void GenericCubemapTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  bind();
  initializeTexture2D();
  shader->setTextureUniforms(location_name, texture_unit);
  unbind();
}

const char *GenericCubemapTexture::getTextureTypeCStr() { return location_name.c_str(); }
