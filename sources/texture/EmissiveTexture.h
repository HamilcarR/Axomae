#ifndef EMISSIVETEXTURE_H
#define EMISSIVETEXTURE_H
#include "GenericTexture.h"

class EmissiveTexture : public GenericTexture {
 protected:
  EmissiveTexture();
  explicit EmissiveTexture(TextureData *data);

 public:
  void initializeTexture2D() override;
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

#endif  // EMISSIVETEXTURE_H
