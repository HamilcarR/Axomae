#ifndef TEXTURE_H
#define TEXTURE_H

#include "GenericTextureProcessing.h"
#include "constants.h"
#include "utils_3D.h"

/**
 * @file Texture.h
 * Implementation of the texture classes
 *
 */

/******************************************************************************************************************************************************************************************************************/
class Shader;
class Texture;

/**
 * @brief Texture class
 *
 */
class Texture {
 protected:
  Texture();
  Texture(TextureData *tex);

 public:
  virtual ~Texture() {}

  /**
   * @brief Internal format of textures
   *
   */
  enum FORMAT : unsigned {
    /*Internal and data formats*/
    RG = GL_RG,
    RGBA = GL_RGBA,       /**<RGBA with 8 bits per channel*/
    BGRA = GL_BGRA,       /**<BGRA with 8 bits per channel*/
    RGB = GL_RGB,         /**<RGB with 8 bits per channel*/
    BGR = GL_BGR,         /**<BGR with 8 bits per channel*/
    RGBA16F = GL_RGBA16F, /**<RGBA with 16 bits floating point per channel*/
    RGBA32F = GL_RGBA32F, /**<RGBA with 32 bits floating point per channel*/
    RGB16F = GL_RGB16F,   /**<RGB with 16 bits floating point per channel*/
    RGB32F = GL_RGB32F,   /**<RGB with 32 bits floating point per channel*/
    /*Data type*/
    UBYTE = GL_UNSIGNED_BYTE, /**<Unsigned byte*/
    FLOAT = GL_FLOAT          /**<4 byte float*/
  };

  enum TYPE : signed {
    GENERIC_CUBE = -3,    /**<Generic cubemap for general purpose */
    GENERIC = -2,         /**<Generic texture used for general purpose */
    EMPTY = -1,           /**<Designate an empty , non generated texture*/
    FRAMEBUFFER = 1,      /**<A texture to be rendered and displayed as a custom
                             framebuffer , by the screen*/
    DIFFUSE = 2,          /**<Diffuse texture. In case the shader used is PBR , this is
                             the albedo*/
    NORMAL = 3,           /**<A normal map texture. Stores normal data*/
    METALLIC = 4,         /**<Metallic texture. Stores the amount of metallic property
                             at a given texel.*/
    ROUGHNESS = 5,        /**<A roughness texture.*/
    AMBIANTOCCLUSION = 6, /**<Ambiant occlusion texture. Occludes light
                             contributions in some areas of the mesh */
    SPECULAR = 7,         /**<Specular texture. In case of PBR , this texture may not be used*/
    EMISSIVE = 8,         /**<Emissive texture. This texture emits light */
    OPACITY = 9,          /**<Alpha blending map . Provides transparency data*/
    CUBEMAP = 10,         /**<Environment map , in the form of a cubemap. Possesses mip
                             maps for use in specular BRDF*/
    ENVMAP2D = 11,        /**<Raw 2D environment map , in equirectangular form. This
                             texture is not used in the final draw loop , until it has
                             been baked into a regular cubemap. */
    IRRADIANCE = 12,      /**<Irradiance map . Provides ambient lighting data to the
                             PBR shaders. */
    BRDFLUT = 13          /**<BRDF lookup texture. Stores reflection factor according to
                             it's texture coordinates*/
  };

  /**
   * @brief Sets the raw data
   * @param texture A pointer to a TextureData object that contains information
   * about the texture, including its width, height, and pixel data.
   */
  virtual void set(TextureData *texture);
  unsigned int getSamplerID() { return sampler2D; }

  /**
   * @brief Set the texture's sampler ID .
   * This method will not check if sampler2D has already a valid value.
   * In this case , the caller needs to free the sampler2D ID first.
   * @param id New Sampler2D id.
   */
  void setSamplerID(unsigned int id) { sampler2D = id; }
  void setTextureType(TYPE type) { name = type; }
  TYPE getTextureType() { return name; };
  virtual bool isDummyTexture() { return is_dummy; }
  void setDummy(bool d) { is_dummy = d; }

  /**
   * @brief Checks if the texture has raw pixel data stored
   *
   */
  virtual bool empty() { return data.empty() && f_data.empty(); }
  virtual bool isInitialized() { return sampler2D != 0; }
  virtual void setMipmapsLevel(unsigned level) { mipmaps = level; }
  virtual unsigned int getMipmapsLevel() { return mipmaps; }
  /**
   * @brief Generate mip maps , and set texture filters accordingly
   * (LINEAR_MIPMAP_LINEAR)
   *
   */
  virtual void generateMipmap();
  virtual void bindTexture() = 0;
  virtual void unbindTexture() = 0;
  virtual void setGlData(Shader *shader) = 0;
  void cleanGlData();
  virtual void setNewSize(unsigned width, unsigned height);
  void clean();

 protected:
  /**
   * @brief Initialize texture filters , mipmaps and glTexImage2D
   *
   */
  virtual void initializeTexture2D();
  virtual void setTextureParametersOptions();

 protected:
  TYPE name;              /**<Type of the texture*/
  FORMAT internal_format; /**<Data layout format on the GPU*/
  FORMAT data_format;     /**<Raw texture data format*/
  FORMAT data_type;
  unsigned int width;         /**<Width of the texture*/
  unsigned int height;        /**<Height of the texture*/
  std::vector<uint32_t> data; /**<Raw data of the texture*/
  std::vector<float> f_data;
  unsigned int sampler2D; /**<ID of the texture*/
  unsigned int mipmaps;   /**<Texture mipmaps level*/
  bool is_dummy;          /**<Check if the current texture is a dummy texture*/
  bool is_initialized;
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Diffuse Texture implementation
 *
 */
class DiffuseTexture : public Texture {
 protected:
  DiffuseTexture();
  DiffuseTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  /**
   * @brief Set the OpenGL texture data infos
   *
   */
  virtual void setGlData(Shader *shader);
  /**
   * @brief This overriden method will additionally check for the presence of
   * transparency in the map. If alpha < 1.f , the texture is considered as
   * having transparency values.
   *
   * @param texture Texture data to copy.
   */
  virtual void set(TextureData *texture) override;

  virtual bool hasTransparency() { return has_transparency; }

  /**
   * @brief Get the texture string description
   *
   * @return C string
   */
  static const char *getTextureTypeCStr();

 protected:
  bool has_transparency;
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Normal texture class definition
 *
 */
class NormalTexture : public Texture {
 protected:
  NormalTexture();
  NormalTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Metallic texture class definition
 *
 */
class MetallicTexture : public Texture {
 protected:
  MetallicTexture();
  MetallicTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Roughness texture class definition
 *
 */
class RoughnessTexture : public Texture {
 protected:
  RoughnessTexture();
  RoughnessTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Ambiant occlusion texture class definition
 *
 */
class AmbiantOcclusionTexture : public Texture {
 protected:
  AmbiantOcclusionTexture();
  AmbiantOcclusionTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Specular texture class definition
 *
 */
class SpecularTexture : public Texture {
 protected:
  SpecularTexture();
  SpecularTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Emissive texture class definition
 *
 */
class EmissiveTexture : public Texture {
 protected:
  EmissiveTexture();
  EmissiveTexture(TextureData *data);

 public:
  void initializeTexture2D() override;
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Opacity texture class definition
 *
 */
class OpacityTexture : public Texture {
 protected:
  OpacityTexture();
  OpacityTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Generic texture class definition
 *
 */
class Generic2DTexture : public Texture {
 protected:
  Generic2DTexture();
  Generic2DTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  virtual void setTextureUnit(unsigned int texture_unit);
  virtual void setLocationName(std::string name);
  const char *getTextureTypeCStr();

 protected:
  unsigned int texture_unit;
  std::string location_name;
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Cubemap texture class definition
 *
 */
class CubemapTexture : public Texture {
 protected:
  CubemapTexture(FORMAT internal_format = RGBA, FORMAT data_format = RGBA, FORMAT data_type = UBYTE, unsigned width = 0, unsigned height = 0);

  CubemapTexture(TextureData *data);

 public:
  /*
   *
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

  virtual void initializeTexture2D() override;
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setNewSize(unsigned _width, unsigned _height) override;
  virtual void setGlData(Shader *shader);
  virtual void generateMipmap() override;
  static const char *getTextureTypeCStr();

 protected:
  virtual void setCubeMapTextureData(TextureData *texture);
};

/******************************************************************************************************************************************************************************************************************/
class GenericCubemapTexture : public CubemapTexture {
 protected:
  GenericCubemapTexture(FORMAT internal_format = RGBA, FORMAT data_format = RGBA, FORMAT data_type = UBYTE, unsigned width = 0, unsigned height = 0);

  GenericCubemapTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  const char *getTextureTypeCStr();

  void setTextureUnit(unsigned int tex_unit) { texture_unit = tex_unit; }

  void setLocationName(std::string loc_name) { location_name = loc_name; }

  unsigned int getTextureUnit() { return texture_unit; }

  std::string getLocationName() { return location_name; }

 protected:
  unsigned int texture_unit;
  std::string location_name;
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Irradiance texture class definition
 *
 */
class IrradianceTexture : public CubemapTexture {
 protected:
  IrradianceTexture(FORMAT internal_format = RGB16F, FORMAT data_format = RGB, FORMAT data_type = FLOAT, unsigned width = 0, unsigned height = 0);

  IrradianceTexture(TextureData *data);

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
  EnvironmentMap2DTexture(
      FORMAT internal_format = RGB32F, FORMAT data_format = RGB, FORMAT data_type = FLOAT, unsigned width = 0, unsigned height = 0);

  EnvironmentMap2DTexture(TextureData *data);

 public:
  virtual void initializeTexture2D() override;
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  static const char *getTextureTypeCStr();
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @class FrameBufferTexture
 * @brief A custom framebuffer's texture for post processing
 *
 */
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
  FrameBufferTexture(TextureData *data);
  FrameBufferTexture(unsigned width, unsigned height);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  virtual void initializeTexture2D() override;
  static const char *getTextureTypeCStr();

 protected:
};

/******************************************************************************************************************************************************************************************************************/

/**
 * @class BRDFLookupTexture
 * @brief PBR BRDF texture .
 * @note Pre baked texture storing the amount of specular reflection for a given
 * triplets of (I , V , R) , where I is the incident light , V is the view
 * direction , and R a roughness value . IE , for (X , Y) being the texture
 * coordinates , Y is a roughness scale , X <==> (I dot V) is the angle betweend
 * the incident light and view direction.
 */
class BRDFLookupTexture : public Texture {
 protected:
  BRDFLookupTexture();
  BRDFLookupTexture(TextureData *data);

 public:
  virtual void bindTexture();
  virtual void unbindTexture();
  virtual void setGlData(Shader *shader);
  virtual void initializeTexture2D() override;
  static const char *getTextureTypeCStr();

 protected:
};

/******************************************************************************************************************************************************************************************************************/

#endif
