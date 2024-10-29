#ifndef EMISSIVETEXTURE_H
#define EMISSIVETEXTURE_H
#include "GenericTexture.h"

#include <internal/macro/project_macros.h>

class EmissiveTexture : public GenericTexture {
 protected:
  explicit EmissiveTexture(std::nullptr_t) : GenericTexture() { set_dummy_TextureData(this); }
  explicit EmissiveTexture(const U32TexData *data) : GenericTexture(data) {}
  explicit EmissiveTexture(const F32TexData *data) : GenericTexture(data) {}

 public:
  void initializeTexture2D() override;
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return EMISSIVE; }
};

#endif  // EMISSIVETEXTURE_H
