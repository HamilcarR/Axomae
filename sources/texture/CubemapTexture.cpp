#include "CubemapTexture.h"
#include "Shader.h"

CubemapTexture::CubemapTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : GenericTexture() {  //! Move arguments to Texture()
  type = CUBEMAP;
  internal_format = _internal_format;
  data_format = _data_format;
  data_type = _data_type;
  width = _width;
  height = _height;
  mipmaps = 0;
}

void CubemapTexture::setCubeMapTextureData(TextureData *texture) {
  clean();
  width = texture->width;
  height = texture->height;
  internal_format = static_cast<GenericTexture::FORMAT>(texture->internal_format);
  data_format = static_cast<GenericTexture::FORMAT>(texture->data_format);
  data_type = static_cast<GenericTexture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
  /* In case the raw data is in RGB-RGBA with 8 bits/channel*/
  if (!texture->data.empty()) {
    data.resize(width * height * 6);
    for (unsigned int i = 0; i < width * height * 6; i++)
      data[i] = texture->data[i];
  }
  /* In case raw data is 4 bytes float / channel */
  if (!texture->f_data.empty()) {
    f_data.resize(width * height * 6 * texture->nb_components);
    for (unsigned i = 0; i < width * height * 6 * texture->nb_components; i++)
      f_data[i] = texture->f_data[i];
  }
}

CubemapTexture::CubemapTexture(TextureData *data) : CubemapTexture() {
  type = CUBEMAP;
  if (data)
    CubemapTexture::setCubeMapTextureData(data);
}

void CubemapTexture::initializeTexture2D() {
  if (!data.empty())
    for (unsigned int i = 1; i <= 6; i++) {
      uint32_t *pointer_to_data = data.data() + (i - 1) * width * height;
      glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (i - 1), 0, internal_format, (int)width, (int)height, 0, data_format, data_type, pointer_to_data);
      errorCheck(__FILE__, __LINE__);
    }
  else
    for (unsigned i = 0; i < 6; i++) {
      glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internal_format, (int)width, (int)height, 0, data_format, data_type, nullptr);
      errorCheck(__FILE__, __LINE__);
    }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void CubemapTexture::generateMipmap() {
  bindTexture();
  glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  unbindTexture();
}

void CubemapTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  bindTexture();
  initializeTexture2D();
  shader->setTextureUniforms(type2str(CUBEMAP), CUBEMAP);
  unbindTexture();
}

void CubemapTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + CUBEMAP);
  glBindTexture(GL_TEXTURE_CUBE_MAP, sampler2D);
}

void CubemapTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + CUBEMAP);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void CubemapTexture::setNewSize(unsigned w, unsigned h) {
  //! Implements this
}

const char *CubemapTexture::getTextureTypeCStr() { return type2str(CUBEMAP); }
