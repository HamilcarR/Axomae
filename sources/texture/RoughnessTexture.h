

#ifndef ROUGHNESSTEXTURE_H
#define ROUGHNESSTEXTURE_H
#include "GenericTexture.h"

class RoughnessTexture : public GenericTexture {
 protected:
  explicit RoughnessTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  [[nodiscard]] TYPE getTextureType() const override { return ROUGHNESS; }
};
#endif  // ROUGHNESSTEXTURE_H
