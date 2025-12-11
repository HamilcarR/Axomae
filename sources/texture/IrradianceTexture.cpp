

#include "IrradianceTexture.h"
#include "Shader.h"
IrradianceTexture::IrradianceTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : CubemapTexture(_internal_format, _data_format, _data_type, _width, _height) {}

void IrradianceTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  bind();
  initializeTexture2D();
  shader->setTextureUniforms(type2str(IRRADIANCE), IRRADIANCE);
  unbind();
}

void IrradianceTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + IRRADIANCE);
  ax_glBindTexture(GL_TEXTURE_CUBE_MAP, sampler2D);
}

void IrradianceTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + IRRADIANCE);
  ax_glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

const char *IrradianceTexture::getTextureTypeCStr() { return type2str(IRRADIANCE); }
