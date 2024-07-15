#ifndef AMBIANTOCCLUSIONTEXTURE_H
#define AMBIANTOCCLUSIONTEXTURE_H

#include "GenericTexture.h"

class Shader;
class AmbiantOcclusionTexture : public GenericTexture {
 protected:
  explicit AmbiantOcclusionTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  [[nodiscard]] TYPE getTextureType() const override { return AMBIANTOCCLUSION; }
};

#endif  // AMBIANTOCCLUSIONTEXTURE_H
