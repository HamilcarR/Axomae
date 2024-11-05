#include "CubemapTexture.h"
#include "Shader.h"

CubemapTexture::CubemapTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : GenericTexture() {
  internal_format = _internal_format;
  data_format = _data_format;
  data_type = _data_type;
  width = _width;
  height = _height;
  mipmaps = 0;
}

void CubemapTexture::setCubeMapTextureData(const F32TexData *texture) {
  if (!texture)
    return;
  clean();
  width = texture->width;
  height = texture->height;
  internal_format = static_cast<GenericTexture::FORMAT>(texture->internal_format);
  data_format = static_cast<GenericTexture::FORMAT>(texture->data_format);
  data_type = static_cast<GenericTexture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
  /* In case raw data is 4 bytes float / channel */
  if (texture->data) {
    f_data = texture->data;
  }
}

void CubemapTexture::setCubeMapTextureData(const U32TexData *texture) {
  if (!texture)
    return;
  clean();
  width = texture->width;
  height = texture->height;
  internal_format = static_cast<GenericTexture::FORMAT>(texture->internal_format);
  data_format = static_cast<GenericTexture::FORMAT>(texture->data_format);
  data_type = static_cast<GenericTexture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
  /* In case the raw data is in RGB-RGBA with 8 bits/channel*/
  if (texture->data) {
    data = texture->data;
  }
}

void CubemapTexture::initializeTexture2D() {
  if (data)
    for (unsigned int i = 1; i <= 6; i++) {
      const uint32_t *pointer_to_data = data + (i - 1) * width * height;
      ax_glTexImage2D(
          GL_TEXTURE_CUBE_MAP_POSITIVE_X + (i - 1), 0, internal_format, (int)width, (int)height, 0, data_format, data_type, pointer_to_data);
    }
  else
    for (unsigned i = 0; i < 6; i++)
      ax_glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internal_format, (int)width, (int)height, 0, data_format, data_type, nullptr);

  ax_glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
  ax_glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
  ax_glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
  ax_glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  ax_glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void CubemapTexture::generateMipmap() {
  bind();
  ax_glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
  ax_glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  unbind();
}

void CubemapTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  bind();
  initializeTexture2D();
  shader->setTextureUniforms(type2str(CUBEMAP), CUBEMAP);
  unbind();
}

void CubemapTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + CUBEMAP);
  ax_glBindTexture(GL_TEXTURE_CUBE_MAP, sampler2D);
}

void CubemapTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + CUBEMAP);
  ax_glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void CubemapTexture::setNewSize(unsigned w, unsigned h) { EMPTY_FUNCBODY }

const char *CubemapTexture::getTextureTypeCStr() { return type2str(CUBEMAP); }
