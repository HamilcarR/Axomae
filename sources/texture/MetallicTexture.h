#ifndef METALLICTEXTURE_H
#define METALLICTEXTURE_H

#include "GenericTexture.h"

class MetallicTexture : public GenericTexture {
 protected:
  explicit MetallicTexture(std::nullptr_t) : GenericTexture() { set_dummy_TextureData(this); }
  explicit MetallicTexture(const U32TexData *data) : GenericTexture(data) {}
  explicit MetallicTexture(const F32TexData *data) : GenericTexture(data) {}

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();

  ax_no_discard TYPE getTextureType() const override { return METALLIC; }
};

#endif  // METALLICTEXTURE_H
