

#ifndef ROUGHNESSTEXTURE_H
#define ROUGHNESSTEXTURE_H
#include "GenericTexture.h"

class RoughnessTexture : public GenericTexture {
 protected:
  RoughnessTexture();
  explicit RoughnessTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};
#endif  // ROUGHNESSTEXTURE_H
