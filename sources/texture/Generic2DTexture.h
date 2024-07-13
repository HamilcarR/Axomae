#ifndef GENERIC2DTEXTURE_H
#define GENERIC2DTEXTURE_H

#include "GenericTexture.h"

class Generic2DTexture : public GenericTexture {
 protected:
  unsigned int texture_unit{};
  std::string location_name;

 protected:
  Generic2DTexture();
  explicit Generic2DTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  virtual void setTextureUnit(unsigned int texture_unit);
  void setLocationName(const std::string &name);
  const char *getTextureTypeCStr();
};

#endif  // GENERIC2DTEXTURE_H
