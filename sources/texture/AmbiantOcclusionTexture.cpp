
#include "AmbiantOcclusionTexture.h"
#include "Shader.h"
AmbiantOcclusionTexture::AmbiantOcclusionTexture() { type = AMBIANTOCCLUSION; }

AmbiantOcclusionTexture::AmbiantOcclusionTexture(TextureData *data) : GenericTexture(data) {
  type = AMBIANTOCCLUSION;
  if (!data)
    set_dummy_TextureData(this);
}

void AmbiantOcclusionTexture::setGlData(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(AMBIANTOCCLUSION), type);
}

void AmbiantOcclusionTexture::bindTexture() {
  ax_glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void AmbiantOcclusionTexture::unbindTexture() {
  ax_glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *AmbiantOcclusionTexture::getTextureTypeCStr() { return type2str(AMBIANTOCCLUSION); }
