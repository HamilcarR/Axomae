

#ifndef NORMALTEXTURE_H
#define NORMALTEXTURE_H
#include "GenericTexture.h"

class NormalTexture : public GenericTexture {
 protected:
  NormalTexture() = default;
  explicit NormalTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return NORMAL; }
};

#endif  // NORMALTEXTURE_H
