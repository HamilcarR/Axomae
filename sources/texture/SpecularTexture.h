

#ifndef SPECULARTEXTURE_H
#define SPECULARTEXTURE_H

#include "GenericTexture.h"

class SpecularTexture : public GenericTexture {
 protected:
  SpecularTexture();
  explicit SpecularTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

#endif  // SPECULARTEXTURE_H
