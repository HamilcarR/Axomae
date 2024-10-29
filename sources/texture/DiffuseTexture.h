#ifndef DIFFUSETEXTURE_H
#define DIFFUSETEXTURE_H
#include "GenericTexture.h"

#include <internal/macro/project_macros.h>

class DiffuseTexture : public GenericTexture {
 protected:
  bool has_transparency{};

 protected:
  explicit DiffuseTexture(std::nullptr_t null_val) { set_dummy_TextureData(this); }
  explicit DiffuseTexture(const U32TexData *data) : GenericTexture(data) {}
  explicit DiffuseTexture(const F32TexData *data) : GenericTexture(data) {}

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
  void set(const U32TexData *texture) override;
  virtual bool hasTransparency() { return has_transparency; }
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return DIFFUSE; }
};

#endif  // DIFFUSETEXTURE_H
