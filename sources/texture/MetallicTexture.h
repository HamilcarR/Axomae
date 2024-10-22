#ifndef METALLICTEXTURE_H
#define METALLICTEXTURE_H

#include "GenericTexture.h"
class TextureData;
class MetallicTexture : public GenericTexture {
 protected:
  explicit MetallicTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();

  ax_no_discard TYPE getTextureType() const override { return METALLIC; }
};

#endif  // METALLICTEXTURE_H
