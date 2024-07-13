//
// Created by hamilcar on 7/13/24.
//

#include "EnvironmentMap2DTexture.h"
#include "Shader.h"

/* This is loaded from the disk as an .hdr image , with 4 bytes float for texel format on each channel...
 * note: make it generic so that we can generate it on the fly ?
 */

EnvironmentMap2DTexture::EnvironmentMap2DTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : GenericTexture() {
  type = ENVMAP2D;
  internal_format = _internal_format;
  data_format = _data_format;
  data_type = _data_type;
  width = _width;
  height = _height;
}

EnvironmentMap2DTexture::EnvironmentMap2DTexture(TextureData *data) : GenericTexture(data) { type = ENVMAP2D; }

void EnvironmentMap2DTexture::initializeTexture2D() {
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, (int)width, (int)height, 0, data_format, data_type, f_data.data());
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glGenerateMipmap(GL_TEXTURE_2D);
  errorCheck(__FILE__, __LINE__);
}

void EnvironmentMap2DTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + ENVMAP2D);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!f_data.empty())
    EnvironmentMap2DTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(type), type);
}

void EnvironmentMap2DTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + ENVMAP2D);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void EnvironmentMap2DTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + ENVMAP2D);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *EnvironmentMap2DTexture::getTextureTypeCStr() { return type2str(ENVMAP2D); }
