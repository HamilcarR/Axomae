#include "DiffuseTexture.h"
#include "Shader.h"

void DiffuseTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(DIFFUSE), DIFFUSE);
}

void DiffuseTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void DiffuseTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

void DiffuseTexture::set(const U32TexData *texture) {
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
