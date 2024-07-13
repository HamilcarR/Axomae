#ifndef OPACITYTEXTURE_H
#define OPACITYTEXTURE_H
#include "GenericTexture.h"
class OpacityTexture : public GenericTexture {
 protected:
  OpacityTexture();
  explicit OpacityTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

#endif  // OPACITYTEXTURE_H
