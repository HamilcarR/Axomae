#ifndef ENVIRONMENTMAP2DTEXTURE_H
#define ENVIRONMENTMAP2DTEXTURE_H

#include "GenericTexture.h"

#include <internal/macro/project_macros.h>

/**
 * @brief Environment map texture class definition
 */
class EnvironmentMap2DTexture : public GenericTexture {
 protected:
  explicit EnvironmentMap2DTexture(
      FORMAT internal_format = RGB32F, FORMAT data_format = RGB, FORMAT data_type = FLOAT, unsigned width = 0, unsigned height = 0);
  explicit EnvironmentMap2DTexture(TextureData *data);

 public:
  void initializeTexture2D() override;
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return ENVMAP2D; }
};

#endif  // ENVIRONMENTMAP2DTEXTURE_H
