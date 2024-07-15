#include "RoughnessTexture.h"
#include "Shader.h"

RoughnessTexture::RoughnessTexture(TextureData *data) : GenericTexture(data) {
  if (!data)
    set_dummy_TextureData(this);
}

void RoughnessTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + ROUGHNESS);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(ROUGHNESS), ROUGHNESS);
}

void RoughnessTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + ROUGHNESS);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void RoughnessTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + ROUGHNESS);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *RoughnessTexture::getTextureTypeCStr() { return type2str(ROUGHNESS); }
