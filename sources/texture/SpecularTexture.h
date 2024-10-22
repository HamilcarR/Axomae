

#ifndef SPECULARTEXTURE_H
#define SPECULARTEXTURE_H

#include "GenericTexture.h"

class SpecularTexture : public GenericTexture {
 protected:
  explicit SpecularTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return SPECULAR; }
};

#endif  // SPECULARTEXTURE_H
