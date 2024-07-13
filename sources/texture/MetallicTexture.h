//
// Created by hamilcar on 7/13/24.
//

#ifndef METALLICTEXTURE_H
#define METALLICTEXTURE_H

#include "GenericTexture.h"
class TextureData;
class MetallicTexture : public GenericTexture {
 protected:
  MetallicTexture();
  explicit MetallicTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

#endif  // METALLICTEXTURE_H
