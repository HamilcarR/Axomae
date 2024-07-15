
#include "BRDFLookupTexture.h"
#include "Shader.h"

BRDFLookupTexture::BRDFLookupTexture(TextureData *data) : GenericTexture(data) {}

void BRDFLookupTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + BRDFLUT);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void BRDFLookupTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + BRDFLUT);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

void BRDFLookupTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  bind();
  initializeTexture2D();
  shader->setTextureUniforms(type2str(BRDFLUT), BRDFLUT);
}

void BRDFLookupTexture::initializeTexture2D() {
  if (!data.empty()) {
    ax_glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, nullptr);
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    ax_glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, f_data.data());
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
}

const char *BRDFLookupTexture::getTextureTypeCStr() { return type2str(BRDFLUT); }
