#include "GenericTexture.h"

static constexpr unsigned int DUMMY_TEXTURE_DIM = 1;
static constexpr uint32_t DEFAULT_NORMAL_DUMMY_PIXEL_RGBA[] = {0x007F7FFF};   // Default pixel color for a normal map
static constexpr uint32_t DEFAULT_OPACITY_DUMMY_PIXEL_RGBA[] = {0xFF000000};  // Default pixel color for other textures

void GenericTexture::setGenericDummy() {
  U32TexData texdata;
  texdata.width = DUMMY_TEXTURE_DIM;
  texdata.height = DUMMY_TEXTURE_DIM;
  texdata.data = reinterpret_cast<const uint32_t *>(DEFAULT_OPACITY_DUMMY_PIXEL_RGBA);
  set(&texdata);
  setDummy(true);
}

void GenericTexture::setNormalmapDummy() {
  U32TexData texdata;
  texdata.width = DUMMY_TEXTURE_DIM;
  texdata.height = DUMMY_TEXTURE_DIM;
  texdata.data = reinterpret_cast<const uint32_t *>(DEFAULT_NORMAL_DUMMY_PIXEL_RGBA);
  set(&texdata);
  setDummy(true);
}

GenericTexture::GenericTexture() {
  is_dummy = false;
  width = 0;
  height = 0;
  sampler2D = 0;
  internal_format = RGBA;
  data_format = BGRA;
}
void GenericTexture::set(const U32TexData *texture) {
  if (!texture)
    return;
  clean();
  name = texture->name;
  width = texture->width;
  height = texture->height;
  if (texture->data)
    data = texture->data;
  data_format = static_cast<GenericTexture::FORMAT>(texture->data_format);
  internal_format = static_cast<GenericTexture::FORMAT>(texture->internal_format);
  data_type = static_cast<GenericTexture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
}

void GenericTexture::set(const F32TexData *texture) {
  if (!texture)
    return;
  clean();
  name = texture->name;
  width = texture->width;
  height = texture->height;
  if (texture->data)
    f_data = texture->data;
  data_format = static_cast<GenericTexture::FORMAT>(texture->data_format);
  internal_format = static_cast<GenericTexture::FORMAT>(texture->internal_format);
  data_type = static_cast<GenericTexture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
}

void GenericTexture::clean() {
  cleanGlData();
  data = nullptr;
  f_data = nullptr;
}

void GenericTexture::setTextureParametersOptions() {
  ax_glGenerateMipmap(GL_TEXTURE_2D);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void GenericTexture::initializeTexture2D() {
  ax_glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, data);
  setTextureParametersOptions();
}

void GenericTexture::generateMipmap() {
  bind();
  ax_glGenerateMipmap(GL_TEXTURE_2D);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  unbind();
}

void GenericTexture::setNewSize(unsigned _width, unsigned _height) {
  width = _width;
  height = _height;
  bind();
  ax_glTexImage2D(GL_TEXTURE_2D, 0, internal_format, _width, _height, 0, data_format, data_type, nullptr);
  unbind();
}

void GenericTexture::cleanGlData() {
  if (sampler2D != 0) {
    glDeleteTextures(1, &sampler2D);
    sampler2D = 0;
  }
}
