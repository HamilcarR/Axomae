#include "GenericTexture.h"
#include "Shader.h"

GenericTexture::GenericTexture() {
  type = EMPTY;
  is_dummy = false;
  width = 0;
  height = 0;
  sampler2D = 0;
  internal_format = RGBA;
  data_format = BGRA;
}

GenericTexture::GenericTexture(TextureData *tex) : GenericTexture() {
  if (tex != nullptr)
    GenericTexture::set(tex);
}

void GenericTexture::set(TextureData *texture) {
  clean();
  name = texture->name;
  width = texture->width;
  height = texture->height;
  if (!texture->data.empty()) {
    data = texture->data;
  }
  if (!texture->f_data.empty()) {
    f_data = texture->f_data;
  }
  data_format = static_cast<GenericTexture::FORMAT>(texture->data_format);
  internal_format = static_cast<GenericTexture::FORMAT>(texture->internal_format);
  data_type = static_cast<GenericTexture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
}

void GenericTexture::clean() { cleanGlData(); }

void GenericTexture::setTextureParametersOptions() {
  GL_ERROR_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
  GL_ERROR_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
  GL_ERROR_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
  GL_ERROR_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
  GL_ERROR_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
}

void GenericTexture::initializeTexture2D() {
  GL_ERROR_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, data.data()));
  setTextureParametersOptions();
}

void GenericTexture::generateMipmap() {
  bindTexture();
  GL_ERROR_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
  GL_ERROR_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
  unbindTexture();
}

void GenericTexture::setNewSize(unsigned _width, unsigned _height) {
  width = _width;
  height = _height;
  bindTexture();
  GL_ERROR_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, internal_format, _width, _height, 0, data_format, data_type, nullptr));
  unbindTexture();
}

void GenericTexture::cleanGlData() {
  if (sampler2D != 0) {
    glDeleteTextures(1, &sampler2D);
    sampler2D = 0;
  }
}
