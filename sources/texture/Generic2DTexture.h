#ifndef GENERIC2DTEXTURE_H
#define GENERIC2DTEXTURE_H

#include "GenericTexture.h"

#include <internal/macro/project_macros.h>

class Generic2DTexture : public GenericTexture {
 protected:
  unsigned int texture_unit{};
  std::string location_name;

 protected:
  explicit Generic2DTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  virtual void setTextureUnit(unsigned int texture_unit);
  void setLocationName(const std::string &name);
  const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return GENERIC; }
};

#endif  // GENERIC2DTEXTURE_H
