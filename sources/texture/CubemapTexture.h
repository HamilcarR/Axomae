#ifndef CUBEMAPTEXTURE_H
#define CUBEMAPTEXTURE_H
#include "GenericTexture.h"

#include <internal/macro/project_macros.h>

class CubemapTexture : public GenericTexture {
 protected:
  explicit CubemapTexture(
      FORMAT internal_format = RGBA, FORMAT data_format = RGBA, FORMAT data_type = UBYTE, unsigned width = 0, unsigned height = 0);
  explicit CubemapTexture(const F32TexData *data) : CubemapTexture() { CubemapTexture::setCubeMapTextureData(data); }
  explicit CubemapTexture(const U32TexData *data) : CubemapTexture() { CubemapTexture::setCubeMapTextureData(data); }
  explicit CubemapTexture(std::nullptr_t) : CubemapTexture() {}

 public:
  /*
   * width * height is the size of one single face. The total size of the
   *cubemap will be :
   *
   * 	6 x width x height x sizeof(uint32_t) bytes
   * with height = width .
   * Here is the layout for mapping the texture :
   *
   *     	  width² = RIGHT => GL_TEXTURE_CUBE_MAP_POSITIVE_X
   * 	  2 x width² = LEFT => GL_TEXTURE_CUBE_MAP_NEGATIVE_X
   * 	  3 x width² = TOP => GL_TEXTURE_CUBE_MAP_POSITIVE_Y
   * 	  4 x width² = BOTTOM => GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
   * 	  5 x width² = BACK => GL_TEXTURE_CUBE_MAP_POSITIVE_Z
   * 	  6 x width² = FRONT => GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
   *!Note : If TextureData == nullptr , this will instead allocate an empty
   *cubemap .
   */

  void initializeTexture2D() override;
  void bind() override;
  void unbind() override;
  void setNewSize(unsigned _width, unsigned _height) override;
  void initialize(Shader *shader) override;
  void generateMipmap() override;
  static const char *getTextureTypeCStr();
  ax_no_discard TYPE getTextureType() const override { return CUBEMAP; }

 protected:
  void setCubeMapTextureData(const U32TexData *texture);
  void setCubeMapTextureData(const F32TexData *texture);
};

#endif  // CUBEMAPTEXTURE_H
