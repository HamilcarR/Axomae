
#include "AmbiantOcclusionTexture.h"
#include "Shader.h"

void AmbiantOcclusionTexture::initialize(Shader *shader) {
  ax_glGenTextures(1, &sampler2D);
  ax_glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (data)
    GenericTexture::initializeTexture2D();
  shader->setTextureUniforms(type2str(AMBIANTOCCLUSION), AMBIANTOCCLUSION);
}

void AmbiantOcclusionTexture::bind() {
  ax_glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  ax_glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void AmbiantOcclusionTexture::unbind() {
  ax_glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  ax_glBindTexture(GL_TEXTURE_2D, 0);
}

const char *AmbiantOcclusionTexture::getTextureTypeCStr() { return type2str(AMBIANTOCCLUSION); }
