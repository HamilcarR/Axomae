
#include "EmissiveTexture.h"
#include "Shader.h"

EmissiveTexture::EmissiveTexture() { type = EMISSIVE; }

EmissiveTexture::EmissiveTexture(TextureData *data) : GenericTexture(data) {
  type = EMISSIVE;
  if (!data)
    set_dummy_TextureData(this);
}

void EmissiveTexture::initializeTexture2D() {
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, data.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  errorCheck(__FILE__, __LINE__);
}

void EmissiveTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    initializeTexture2D();
  shader->setTextureUniforms(type2str(type), type);
}

void EmissiveTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void EmissiveTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *EmissiveTexture::getTextureTypeCStr() { return type2str(EMISSIVE); }
