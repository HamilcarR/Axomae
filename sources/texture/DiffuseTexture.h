//
// Created by hamilcar on 7/13/24.
//

#ifndef DIFFUSETEXTURE_H
#define DIFFUSETEXTURE_H
#include "GenericTexture.h"

class DiffuseTexture : public GenericTexture {
 protected:
  bool has_transparency;

 protected:
  DiffuseTexture();
  explicit DiffuseTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  /**
   * @brief Set the OpenGL texture data infos
   */
  void setGlData(Shader *shader) override;
  /**
   * @brief This overriden method will additionally check for the presence of
   * transparency in the map. If alpha < 1.f , the texture is considered as
   * having transparency values.
   * @param texture Texture data to copy.
   */
  void set(TextureData *texture) override;
  virtual bool hasTransparency() { return has_transparency; }
  static const char *getTextureTypeCStr();
};

#endif  // DIFFUSETEXTURE_H
