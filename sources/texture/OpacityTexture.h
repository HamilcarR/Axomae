#ifndef OPACITYTEXTURE_H
#define OPACITYTEXTURE_H
#include "GenericTexture.h"
class OpacityTexture : public GenericTexture {
 protected:
  explicit OpacityTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return OPACITY; }
};

#endif  // OPACITYTEXTURE_H
