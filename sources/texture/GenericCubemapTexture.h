

#ifndef GENERICCUBEMAPTEXTURE_H
#define GENERICCUBEMAPTEXTURE_H

#include "CubemapTexture.h"

class GenericCubemapTexture : public CubemapTexture {
 protected:
  unsigned int texture_unit;
  std::string location_name;

 protected:
  explicit GenericCubemapTexture(
      FORMAT internal_format = RGBA, FORMAT data_format = RGBA, FORMAT data_type = UBYTE, unsigned width = 0, unsigned height = 0);
  explicit GenericCubemapTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  const char *getTextureTypeCStr();
  void setTextureUnit(unsigned int tex_unit) { texture_unit = tex_unit; }
  void setLocationName(const std::string &loc_name) { location_name = loc_name; }
  [[nodiscard]] unsigned int getTextureUnit() const { return texture_unit; }
  std::string getLocationName() { return location_name; }
  [[nodiscard]] TYPE getTextureType() const override { return GENERIC_CUBE; }
};

#endif  // GENERICCUBEMAPTEXTURE_H
