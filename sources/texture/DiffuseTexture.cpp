#include "DiffuseTexture.h"
#include "Shader.h"

DiffuseTexture::DiffuseTexture() { type = DIFFUSE; }

DiffuseTexture::DiffuseTexture(TextureData *data) : GenericTexture(data) {
  type = DIFFUSE;
  if (!data)
    set_dummy_TextureData(this);
}

void DiffuseTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(DIFFUSE), DIFFUSE);
}

void DiffuseTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void DiffuseTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void DiffuseTexture::set(TextureData *texture) {
  clean();
  width = texture->width;
  height = texture->height;
  data.resize(width * height);
  data_format = static_cast<GenericTexture::FORMAT>(texture->data_format);
  internal_format = static_cast<GenericTexture::FORMAT>(texture->internal_format);
  data_type = static_cast<GenericTexture::FORMAT>(texture->data_type);
  has_transparency = false;
  for (unsigned int i = 0; i < width * height; i++) {
    data[i] = texture->data[i];
    if ((data[i] & 0xFF000000) != 0xFF000000)  // check for transparency
      has_transparency = true;
  }
}

const char *DiffuseTexture::getTextureTypeCStr() { return type2str(DIFFUSE); }
