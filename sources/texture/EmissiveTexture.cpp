
#include "EmissiveTexture.h"
#include "Shader.h"

EmissiveTexture::EmissiveTexture(TextureData *data) : GenericTexture(data) {
  if (!data)
    set_dummy_TextureData(this);
}

void EmissiveTexture::initializeTexture2D() {
  ax_glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, data.data());
  ax_glGenerateMipmap(GL_TEXTURE_2D);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void EmissiveTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    initializeTexture2D();
  shader->setTextureUniforms(type2str(EMISSIVE), EMISSIVE);
}

void EmissiveTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void EmissiveTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *EmissiveTexture::getTextureTypeCStr() { return type2str(EMISSIVE); }
