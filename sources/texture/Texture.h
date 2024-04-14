#ifndef TEXTURE_H
#define TEXTURE_H

#include "GenericTextureProcessing.h"
#include "constants.h"
#include "init_3D.h"
#include "utils_3D.h"

/**
 * @file Texture.h
 */

class Shader;
class Texture;

/**
 * @brief Texture class
 */
class Texture {
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
  Texture();
  explicit Texture(TextureData *tex);

 public:
  virtual ~Texture() = default;
  Texture(const Texture &copy) = default;
  Texture(Texture &&move) noexcept = default;
  Texture &operator=(const Texture &copy) = default;
  Texture &operator=(Texture &&move) noexcept = default;
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
void Texture::setNewData(const std::vector<T> &new_buffer, FORMAT format, FORMAT type) {
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (int)width, (int)height, format, type, new_buffer.data());
}
/******************************************************************************************************************************************************************************************************************/
class DiffuseTexture : public Texture {
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

/******************************************************************************************************************************************************************************************************************/
class NormalTexture : public Texture {
 protected:
  NormalTexture();
  explicit NormalTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class MetallicTexture : public Texture {
 protected:
  MetallicTexture();
  explicit MetallicTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class RoughnessTexture : public Texture {
 protected:
  RoughnessTexture();
  explicit RoughnessTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class AmbiantOcclusionTexture : public Texture {
 protected:
  AmbiantOcclusionTexture();
  explicit AmbiantOcclusionTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class SpecularTexture : public Texture {
 protected:
  SpecularTexture();
  explicit SpecularTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class EmissiveTexture : public Texture {
 protected:
  EmissiveTexture();
  explicit EmissiveTexture(TextureData *data);

 public:
  void initializeTexture2D() override;
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class OpacityTexture : public Texture {
 protected:
  OpacityTexture();
  explicit OpacityTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class Generic2DTexture : public Texture {
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

/******************************************************************************************************************************************************************************************************************/
class CubemapTexture : public Texture {
 protected:
  explicit CubemapTexture(
      FORMAT internal_format = RGBA, FORMAT data_format = RGBA, FORMAT data_type = UBYTE, unsigned width = 0, unsigned height = 0);
  explicit CubemapTexture(TextureData *data);

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
  void bindTexture() override;
  void unbindTexture() override;
  void setNewSize(unsigned _width, unsigned _height) override;
  void setGlData(Shader *shader) override;
  void generateMipmap() override;
  static const char *getTextureTypeCStr();

 protected:
  virtual void setCubeMapTextureData(TextureData *texture);
};

/******************************************************************************************************************************************************************************************************************/
class GenericCubemapTexture : public CubemapTexture {
 protected:
  unsigned int texture_unit;
  std::string location_name;

 protected:
  explicit GenericCubemapTexture(
      FORMAT internal_format = RGBA, FORMAT data_format = RGBA, FORMAT data_type = UBYTE, unsigned width = 0, unsigned height = 0);
  explicit GenericCubemapTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  const char *getTextureTypeCStr();
  void setTextureUnit(unsigned int tex_unit) { texture_unit = tex_unit; }
  void setLocationName(const std::string &loc_name) { location_name = loc_name; }
  [[nodiscard]] unsigned int getTextureUnit() const { return texture_unit; }
  std::string getLocationName() { return location_name; }
};

/******************************************************************************************************************************************************************************************************************/
class IrradianceTexture : public CubemapTexture {
 protected:
  explicit IrradianceTexture(
      FORMAT internal_format = RGB16F, FORMAT data_format = RGB, FORMAT data_type = FLOAT, unsigned width = 0, unsigned height = 0);
  explicit IrradianceTexture(TextureData *data);

 public:
  const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Environment map texture class definition
 *
 */
class EnvironmentMap2DTexture : public Texture {
 protected:
  explicit EnvironmentMap2DTexture(
      FORMAT internal_format = RGB32F, FORMAT data_format = RGB, FORMAT data_type = FLOAT, unsigned width = 0, unsigned height = 0);
  explicit EnvironmentMap2DTexture(TextureData *data);

 public:
  void initializeTexture2D() override;
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
class FrameBufferTexture : public Texture {
 protected:
  FrameBufferTexture();
  /**
   * @brief Construct a new Frame Buffer Texture
   * Contains only width , and height... The rest of the TextureData parameter
   * is not used,
   * @param data TextureData parameter
   *
   */
  explicit FrameBufferTexture(TextureData *data);
  FrameBufferTexture(unsigned width, unsigned height);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  void initializeTexture2D() override;
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/

/**
 * @class BRDFLookupTexture
 * @brief PBR BRDF texture .
 * @note Pre baked texture storing the amount of specular reflection for a given
 * triplets of (I , V , R) , where I is the incident light , V is the view
 * direction , and R a roughness value . IE , for (X , Y) being the texture
 * coordinates , Y is a roughness scale , X <==> (I dot V) is the angle between
 * the incident light and view direction.
 */
class BRDFLookupTexture : public Texture {
 protected:
  BRDFLookupTexture();
  explicit BRDFLookupTexture(TextureData *data);

 public:
  void bindTexture() override;
  void unbindTexture() override;
  void setGlData(Shader *shader) override;
  void initializeTexture2D() override;
  static const char *getTextureTypeCStr();
};

#endif
