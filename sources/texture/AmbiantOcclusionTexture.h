#ifndef AMBIANTOCCLUSIONTEXTURE_H
#define AMBIANTOCCLUSIONTEXTURE_H

#include "GenericTexture.h"

#include <internal/macro/project_macros.h>

class Shader;
class AmbiantOcclusionTexture : public GenericTexture {
 protected:
  explicit AmbiantOcclusionTexture(std::nullptr_t) { set_dummy_TextureData(this); }
  explicit AmbiantOcclusionTexture(const U32TexData *data) : GenericTexture(data) {}
  explicit AmbiantOcclusionTexture(const F32TexData *data) : GenericTexture(data) {}

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return AMBIANTOCCLUSION; }
};

#endif  // AMBIANTOCCLUSIONTEXTURE_H
