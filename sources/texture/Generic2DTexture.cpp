#include "Generic2DTexture.h"
#include "Shader.h"

void Generic2DTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + texture_unit);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (data)
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(location_name, texture_unit);
}

void Generic2DTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + texture_unit);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void Generic2DTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + texture_unit);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *Generic2DTexture::getTextureTypeCStr() { return location_name.c_str(); }

void Generic2DTexture::setTextureUnit(unsigned int tex_unit) { texture_unit = tex_unit; }

void Generic2DTexture::setLocationName(const std::string &_name) { location_name = _name; }
