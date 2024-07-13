#ifndef GENERICTEXTURE_H
#define GENERICTEXTURE_H

#include "GenericTextureProcessing.h"
#include "constants.h"
#include "init_3D.h"
#include "utils_3D.h"
#include <map>
#include <unordered_map>

class Shader;
/**
 * @brief Texture class
 */
class GenericTexture {
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
  TYPE type{};
  FORMAT internal_format{};
  FORMAT data_format{};
  FORMAT data_type{};
  unsigned int width{};
  unsigned int height{};
  std::vector<uint32_t> data{};
  std::vector<float> f_data{};
  unsigned int sampler2D{};
  unsigned int mipmaps{};
  bool is_dummy{};
  bool is_initialized{};

 protected:
  GenericTexture();
  explicit GenericTexture(TextureData *tex);

 public:
  virtual ~GenericTexture() = default;
  GenericTexture(const GenericTexture &copy) = default;
  GenericTexture(GenericTexture &&move) noexcept = default;
  GenericTexture &operator=(const GenericTexture &copy) = default;
  GenericTexture &operator=(GenericTexture &&move) noexcept = default;
  virtual void set(TextureData *texture);
  [[nodiscard]] unsigned int getSamplerID() const { return sampler2D; }
  /**
   * @brief Set the texture's sampler ID .
   * This method will not check if sampler2D has already a valid value.
   * In this case , the caller needs to free the sampler2D ID first.
   * This method will also not free or tamper with the previous sampler2D value.
   */
  void setSamplerID(unsigned int id) { sampler2D = id; }
  void setTextureType(TYPE type_) { type = type_; }
  TYPE getTextureType() { return type; };
  [[nodiscard]] const uint32_t *getData() const { return data.data(); }
  [[nodiscard]] const float *getFData() const { return f_data.data(); }
  [[nodiscard]] const unsigned getWidth() const { return width; }
  [[nodiscard]] const unsigned getHeight() const { return height; }
  virtual bool isDummyTexture() { return is_dummy; }
  void setDummy(bool d) { is_dummy = d; }
  [[nodiscard]] const std::string &getName() const { return name; }
  virtual bool empty() { return data.empty() && f_data.empty(); }
  virtual bool isInitialized() { return sampler2D != 0; }
  virtual void setMipmapsLevel(unsigned level) { mipmaps = level; }
  virtual unsigned int getMipmapsLevel() { return mipmaps; }
  virtual void generateMipmap();
  virtual void bindTexture() = 0;
  virtual void unbindTexture() = 0;
  virtual void setGlData(Shader *shader) = 0;
  void cleanGlData();
  virtual void setNewSize(unsigned width, unsigned height);
  template<class T>
  void setNewData(const std::vector<T> &new_buffer, FORMAT format, FORMAT type);
  void clean();

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

constexpr unsigned int DUMMY_TEXTURE_DIM = 1;
constexpr uint32_t DEFAULT_NORMAL_DUMMY_PIXEL_RGBA = 0x007F7FFF;   // Default pixel color for a normal map
constexpr uint32_t DEFAULT_OPACITY_DUMMY_PIXEL_RGBA = 0xFF000000;  // Default pixel color for other textures

inline constexpr const char *type2str(GenericTexture::TYPE type) {
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

/**
 * The function sets the data of a dummy texture to a solid black color.
 * @param dummy The parameter "dummy" is a pointer to a struct of type "TextureData". The function sets
 * the width, height, and data of the TextureData struct pointed to by "dummy" to create a dummy
 * texture with a solid black color.
 */
inline void set_dummy_TextureData(GenericTexture *set_texture) {
  TextureData dummy;
  dummy.width = DUMMY_TEXTURE_DIM;
  dummy.height = DUMMY_TEXTURE_DIM;
  dummy.data.resize(dummy.width * dummy.height);
  for (unsigned i = 0; i < dummy.width * dummy.height; i++) {
    dummy.data[i] = DEFAULT_OPACITY_DUMMY_PIXEL_RGBA;
  }
  set_texture->set(&dummy);
  set_texture->setDummy(true);
}

/**
 * The function sets the width, height, and data of a dummy texture with a constant blue color ,
 * indicating a uniform vector (0 , 0 , 1) representing the normal.
 * @param dummy A pointer to a TextureData struct that will be filled with one only pixel having the value (128 , 128 ,
 * 255).
 */

inline void set_dummy_TextureData_normals(GenericTexture *set_texture) {
  TextureData dummy;
  dummy.width = DUMMY_TEXTURE_DIM;
  dummy.height = DUMMY_TEXTURE_DIM;
  dummy.data.resize(dummy.width * dummy.height);
  for (unsigned i = 0; i < DUMMY_TEXTURE_DIM * DUMMY_TEXTURE_DIM; i++)
    dummy.data[i] = DEFAULT_NORMAL_DUMMY_PIXEL_RGBA;
  set_texture->set(&dummy);
  set_texture->setDummy(true);
}

#endif  // GENERICTEXTURE_H
