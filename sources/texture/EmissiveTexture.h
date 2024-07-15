#ifndef EMISSIVETEXTURE_H
#define EMISSIVETEXTURE_H
#include "GenericTexture.h"

class EmissiveTexture : public GenericTexture {
 protected:
  explicit EmissiveTexture(TextureData *data);

 public:
  void initializeTexture2D() override;
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  [[nodiscard]] TYPE getTextureType() const override { return EMISSIVE; }
};

#endif  // EMISSIVETEXTURE_H
