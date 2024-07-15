#ifndef DIFFUSETEXTURE_H
#define DIFFUSETEXTURE_H
#include "GenericTexture.h"

class DiffuseTexture : public GenericTexture {
 protected:
  bool has_transparency{};

 protected:
  explicit DiffuseTexture(TextureData *data);

 public:
  void bind() override;
  void unbind() override;
  /**
   * @brief Set the OpenGL texture data infos
   */
  void initialize(Shader *shader) override;
  /**
   * @brief This overriden method will additionally check for the presence of
   * transparency in the map. If alpha < 1.f , the texture is considered as
   * having transparency values.
   * @param texture Texture data to copy.
   */
  void set(TextureData *texture) override;
  virtual bool hasTransparency() { return has_transparency; }
  static const char *getTextureTypeCStr();
  [[nodiscard]] TYPE getTextureType() const override { return DIFFUSE; }
};

#endif  // DIFFUSETEXTURE_H
