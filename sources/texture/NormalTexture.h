//
// Created by hamilcar on 7/13/24.
//

#ifndef NORMALTEXTURE_H
#define NORMALTEXTURE_H
#include "GenericTexture.h"

class NormalTexture : public GenericTexture {
 protected:
  NormalTexture();
  explicit NormalTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

#endif  // NORMALTEXTURE_H
