

#ifndef NORMALTEXTURE_H
#define NORMALTEXTURE_H
#include "GenericTexture.h"

class NormalTexture : public GenericTexture {
 protected:
  NormalTexture() = default;
  explicit NormalTexture(std::nullptr_t) : GenericTexture() { set_dummy_TextureData_normals(this); }
  explicit NormalTexture(const U32TexData *texture) : GenericTexture(texture) {}
  explicit NormalTexture(const F32TexData *texture) : GenericTexture(texture) {}

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return NORMAL; }
};

#endif  // NORMALTEXTURE_H
