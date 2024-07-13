
#include "BRDFLookupTexture.h"
#include "Shader.h"
BRDFLookupTexture::BRDFLookupTexture() : GenericTexture() { type = BRDFLUT; }

BRDFLookupTexture::BRDFLookupTexture(TextureData *data) : GenericTexture(data) { type = BRDFLUT; }

void BRDFLookupTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + BRDFLUT);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void BRDFLookupTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + BRDFLUT);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void BRDFLookupTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  bindTexture();
  initializeTexture2D();
  shader->setTextureUniforms(type2str(BRDFLUT), BRDFLUT);
}

void BRDFLookupTexture::initializeTexture2D() {
  if (!data.empty()) {
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, f_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
}

const char *BRDFLookupTexture::getTextureTypeCStr() { return type2str(BRDFLUT); }
