#ifndef GENERICTEXTURE_H
#define GENERICTEXTURE_H

#include "GenericTextureProcessing.h"
#include "internal/device/rendering/DeviceTextureInterface.h"
#include "internal/device/rendering/opengl/init_3D.h"
#include <cstring>
#include <internal/macro/project_macros.h>
class Shader;

/**
 * @brief Texture class
 */
class GenericTexture : public DeviceTextureInterface {
 public:
  enum FORMAT : unsigned {
    /*Internal and data formats*/
    RG = GL_RG,
    RGBA = GL_RGBA,
    BGRA = GL_BGRA,
    RGB = GL_RGB,
    BGR = GL_BGR,
    RGBA16F = GL_RGBA16F,
    RGBA32F = GL_RGBA32F,
    RGB16F = GL_RGB16F,
    RGB32F = GL_RGB32F,
    /*Data type*/
    UBYTE = GL_UNSIGNED_BYTE,
    FLOAT = GL_FLOAT
  };

  enum TYPE : signed {
    GENERIC_CUBE = -3,
    GENERIC = -2,
    EMPTY = -1,
    FRAMEBUFFER = 1,
    DIFFUSE = 2,
    NORMAL = 3,
    METALLIC = 4,
    ROUGHNESS = 5,
    AMBIANTOCCLUSION = 6,
    SPECULAR = 7,
    EMISSIVE = 8,
    OPACITY = 9,
    CUBEMAP = 10,
    ENVMAP2D = 11,
    IRRADIANCE = 12,
    BRDFLUT = 13
  };

 protected:
  std::string name{};
  FORMAT internal_format{};
  FORMAT data_format{};
  FORMAT data_type{};
  unsigned int width{};
  unsigned int height{};
  const uint32_t *data{};
  const float *f_data{};
  unsigned int sampler2D{};
  unsigned int mipmaps{};
  bool is_dummy{};
  bool is_initialized{};

 protected:
  GenericTexture();
  explicit GenericTexture(const U32TexData *tex) { GenericTexture::set(tex); }
  explicit GenericTexture(const F32TexData *tex) { GenericTexture::set(tex); }
  explicit GenericTexture(std::nullptr_t) : GenericTexture() {}

 public:
  ~GenericTexture() override = default;
  GenericTexture(const GenericTexture &copy) = default;
  GenericTexture(GenericTexture &&move) noexcept = default;
  GenericTexture &operator=(const GenericTexture &copy) = default;
  GenericTexture &operator=(GenericTexture &&move) noexcept = default;
  virtual void set(const F32TexData *texture);
  virtual void set(const U32TexData *texture);
  ax_no_discard unsigned int getSamplerID() const { return sampler2D; }
  /**
   * @brief Set the texture's sampler ID .
   * This method will not check if sampler2D has already a valid value.
   * In this case , the caller needs to free the sampler2D ID first.
   * This method will also not free or tamper with the previous sampler2D value.
   */
  void setSamplerID(unsigned int id) { sampler2D = id; }
  ax_no_discard virtual TYPE getTextureType() const { return EMPTY; };
  ax_no_discard const uint32_t *getData() const { return data; }
  ax_no_discard const float *getFData() const { return f_data; }
  ax_no_discard unsigned getWidth() const { return width; }
  ax_no_discard unsigned getHeight() const { return height; }
  ax_no_discard bool isDummyTexture() const { return is_dummy; }
  void setDummy(bool d) { is_dummy = d; }
  void setNormalmapDummy();
  void setGenericDummy();
  ax_no_discard const std::string &getName() const { return name; }
  virtual bool empty() { return !data && !f_data; }
  ax_no_discard bool isInitialized() const override { return sampler2D != 0; }
  virtual void setMipmapsLevel(unsigned level) { mipmaps = level; }
  virtual unsigned int getMipmapsLevel() { return mipmaps; }
  virtual void generateMipmap();
  virtual void bind() = 0;
  virtual void unbind() = 0;
  virtual void initialize(Shader *shader) = 0;
  void cleanGlData();
  virtual void setNewSize(unsigned width, unsigned height);
  template<class T>
  void setNewData(const std::vector<T> &new_buffer, FORMAT format, FORMAT type);
  void clean() override;

 protected:
  /**
   * @brief Initialize texture filters , mipmaps and glTexImage2D
   */
  virtual void initializeTexture2D();
  virtual void setTextureParametersOptions();
};

template<class T>
void GenericTexture::setNewData(const std::vector<T> &new_buffer, FORMAT format, FORMAT type) {
  ax_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (int)width, (int)height, format, type, new_buffer.data());
}

constexpr const char *type2str(GenericTexture::TYPE type) {
  switch (type) {
    case GenericTexture::DIFFUSE:
      return "diffuse_map";
    case GenericTexture::NORMAL:
      return "normal_map";
    case GenericTexture::METALLIC:
      return "metallic_map";
    case GenericTexture::ROUGHNESS:
      return "roughness_map";
    case GenericTexture::AMBIANTOCCLUSION:
      return "ambiantocclusion_map";
    case GenericTexture::SPECULAR:
      return "specular_map";
    case GenericTexture::EMISSIVE:
      return "emissive_map";
    case GenericTexture::OPACITY:
      return "opacity_map";
    case GenericTexture::CUBEMAP:
      return "cubemap";
    case GenericTexture::ENVMAP2D:
      return "environment_map";
    case GenericTexture::IRRADIANCE:
      return "irradiance_map";
    case GenericTexture::GENERIC:
      return "generic_map";
    case GenericTexture::FRAMEBUFFER:
      return "framebuffer_map";
    case GenericTexture::BRDFLUT:
      return "brdf_lookup_map";
    default:
      return "unknown_map";
  }
}

inline GenericTexture::TYPE str2type(const char *str) {
  if (std::strcmp(str, "diffuse_map") == 0) {
    return GenericTexture::DIFFUSE;
  } else if (std::strcmp(str, "normal_map") == 0) {
    return GenericTexture::NORMAL;
  } else if (std::strcmp(str, "metallic_map") == 0) {
    return GenericTexture::METALLIC;
  } else if (std::strcmp(str, "roughness_map") == 0) {
    return GenericTexture::ROUGHNESS;
  } else if (std::strcmp(str, "ambiantocclusion_map") == 0) {
    return GenericTexture::AMBIANTOCCLUSION;
  } else if (std::strcmp(str, "specular_map") == 0) {
    return GenericTexture::SPECULAR;
  } else if (std::strcmp(str, "emissive_map") == 0) {
    return GenericTexture::EMISSIVE;
  } else if (std::strcmp(str, "opacity_map") == 0) {
    return GenericTexture::OPACITY;
  } else if (std::strcmp(str, "cubemap") == 0) {
    return GenericTexture::CUBEMAP;
  } else if (std::strcmp(str, "environment_map") == 0) {
    return GenericTexture::ENVMAP2D;
  } else if (std::strcmp(str, "irradiance_map") == 0) {
    return GenericTexture::IRRADIANCE;
  } else if (std::strcmp(str, "generic_map") == 0) {
    return GenericTexture::GENERIC;
  } else if (std::strcmp(str, "framebuffer_map") == 0) {
    return GenericTexture::FRAMEBUFFER;
  } else if (std::strcmp(str, "brdf_lookup_map") == 0) {
    return GenericTexture::BRDFLUT;
  } else {
    return GenericTexture::EMPTY;
  }
}

/**
 * The function sets the data of a dummy texture to a solid black color.
 * @param dummy The parameter "dummy" is a pointer to a struct of type "TextureData". The function sets
 * the width, height, and data of the TextureData struct pointed to by "dummy" to create a dummy
 * texture with a solid black color.
 */
inline void set_dummy_TextureData(GenericTexture *set_texture) { set_texture->setGenericDummy(); }

/**
 * The function sets the width, height, and data of a dummy texture with a constant blue color ,
 * indicating a uniform vector (0 , 0 , 1) representing the normal.
 * @param dummy A pointer to a TextureData struct that will be filled with one only pixel having the value (128 , 128 ,
 * 255).
 */

inline void set_dummy_TextureData_normals(GenericTexture *set_texture) { set_texture->setNormalmapDummy(); }

#endif  // GENERICTEXTURE_H
