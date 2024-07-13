#ifndef AMBIANTOCCLUSIONTEXTURE_H
#define AMBIANTOCCLUSIONTEXTURE_H

#include "GenericTexture.h"

class Shader;
class AmbiantOcclusionTexture : public GenericTexture {
 protected:
  AmbiantOcclusionTexture();
  explicit AmbiantOcclusionTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

#endif  // AMBIANTOCCLUSIONTEXTURE_H
