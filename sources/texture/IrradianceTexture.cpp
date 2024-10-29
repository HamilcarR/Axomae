

#include "IrradianceTexture.h"

IrradianceTexture::IrradianceTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : CubemapTexture(_internal_format, _data_format, _data_type, _width, _height) {}

const char *IrradianceTexture::getTextureTypeCStr() { return type2str(IRRADIANCE); }
