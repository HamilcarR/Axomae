

#ifndef IRRADIANCETEXTURE_H
#define IRRADIANCETEXTURE_H

#include "CubemapTexture.h"

class IrradianceTexture : public CubemapTexture {
 protected:
  explicit IrradianceTexture(
      FORMAT internal_format = RGB16F, FORMAT data_format = RGB, FORMAT data_type = FLOAT, unsigned width = 0, unsigned height = 0);
  explicit IrradianceTexture(std::nullptr_t) : IrradianceTexture() {}
  explicit IrradianceTexture(const U32TexData *data) : IrradianceTexture() { CubemapTexture::setCubeMapTextureData(data); }
  explicit IrradianceTexture(const F32TexData *data) : IrradianceTexture() { CubemapTexture::setCubeMapTextureData(data); }

 public:
  const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return IRRADIANCE; }
};

#endif  // IRRADIANCETEXTURE_H
