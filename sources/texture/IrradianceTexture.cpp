//
// Created by hamilcar on 7/13/24.
//

#include "IrradianceTexture.h"
#include "Shader.h"

IrradianceTexture::IrradianceTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : CubemapTexture(_internal_format, _data_format, _data_type, _width, _height) {
  type = IRRADIANCE;
}

IrradianceTexture::IrradianceTexture(TextureData *data) : IrradianceTexture() {
  type = IRRADIANCE;
  if (data)
    CubemapTexture::setCubeMapTextureData(data);
}

const char *IrradianceTexture::getTextureTypeCStr() { return type2str(IRRADIANCE); }
