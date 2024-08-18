

#include "IrradianceTexture.h"

IrradianceTexture::IrradianceTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : CubemapTexture(_internal_format, _data_format, _data_type, _width, _height) {}

IrradianceTexture::IrradianceTexture(TextureData *data) : IrradianceTexture() {
  if (data)
    CubemapTexture::setCubeMapTextureData(data);
}

const char *IrradianceTexture::getTextureTypeCStr() { return type2str(IRRADIANCE); }
