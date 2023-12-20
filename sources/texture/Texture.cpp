#include "Texture.h"
#include "DebugGL.h"
#include "Shader.h"
#include <map>

constexpr unsigned int DUMMY_TEXTURE_DIM = 1;
constexpr uint32_t DEFAULT_NORMAL_DUMMY_PIXEL_RGBA = 0x007F7FFF;   // Default pixel color for a normal map
constexpr uint32_t DEFAULT_OPACITY_DUMMY_PIXEL_RGBA = 0xFF000000;  // Default pixel color for other textures
static std::map<Texture::TYPE, const char *> texture_type_c_str = {{Texture::DIFFUSE, "diffuse_map"},
                                                                   {Texture::NORMAL, "normal_map"},
                                                                   {Texture::METALLIC, "metallic_map"},
                                                                   {Texture::ROUGHNESS, "roughness_map"},
                                                                   {Texture::AMBIANTOCCLUSION, "ambiantocclusion_map"},
                                                                   {Texture::SPECULAR, "specular_map"},
                                                                   {Texture::EMISSIVE, "emissive_map"},
                                                                   {Texture::OPACITY, "opacity_map"},
                                                                   {Texture::CUBEMAP, "cubemap"},
                                                                   {Texture::ENVMAP2D, "environment_map"},
                                                                   {Texture::IRRADIANCE, "irradiance_map"},
                                                                   {Texture::GENERIC, "generic_map"},
                                                                   {Texture::FRAMEBUFFER, "framebuffer_map"},
                                                                   {Texture::BRDFLUT, "brdf_lookup_map"}

};

/**
 * The function sets the data of a dummy texture to a solid black color.
 *
 * @param dummy The parameter "dummy" is a pointer to a struct of type "TextureData". The function sets
 * the width, height, and data of the TextureData struct pointed to by "dummy" to create a dummy
 * texture with a solid black color.
 */
static void set_dummy_TextureData(Texture *set_texture) {
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
 *
 * @param dummy A pointer to a TextureData struct that will be filled with one only pixel having the value (128 , 128 ,
 * 255).
 */

static void set_dummy_TextureData_normals(Texture *set_texture) {
  TextureData dummy;
  dummy.width = DUMMY_TEXTURE_DIM;
  dummy.height = DUMMY_TEXTURE_DIM;
  dummy.data.resize(dummy.width * dummy.height);
  for (unsigned i = 0; i < DUMMY_TEXTURE_DIM * DUMMY_TEXTURE_DIM; i++)
    dummy.data[i] = DEFAULT_NORMAL_DUMMY_PIXEL_RGBA;
  set_texture->set(&dummy);
  set_texture->setDummy(true);
}

Texture::Texture() {
  name = EMPTY;
  is_dummy = false;
  width = 0;
  height = 0;
  sampler2D = 0;
  internal_format = RGBA;
  data_format = BGRA;
}

Texture::Texture(TextureData *tex) : Texture() {
  if (tex != nullptr)
    set(tex);
}

void Texture::set(TextureData *texture) {
  clean();
  width = texture->width;
  height = texture->height;
  if (!texture->data.empty()) {
    data.resize(width * height);
    data = texture->data;
  }
  if (!texture->f_data.empty()) {
    f_data.resize(width * height * texture->nb_components);
    f_data = texture->f_data;
  }
  data_format = static_cast<Texture::FORMAT>(texture->data_format);
  internal_format = static_cast<Texture::FORMAT>(texture->internal_format);
  data_type = static_cast<Texture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
}

void Texture::clean() { cleanGlData(); }

void Texture::setTextureParametersOptions() {
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  errorCheck(__FILE__, __LINE__);
}

void Texture::initializeTexture2D() {
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, data.data());
  setTextureParametersOptions();
}

void Texture::generateMipmap() {
  bindTexture();
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  unbindTexture();
}

void Texture::setNewSize(unsigned _width, unsigned _height) {
  width = _width;
  height = _height;
  bindTexture();
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, _width, _height, 0, data_format, data_type, nullptr);
  unbindTexture();
}

void Texture::cleanGlData() {
  if (sampler2D != 0) {
    glDeleteTextures(1, &sampler2D);
    sampler2D = 0;
  }
}

/****************************************************************************************************************************/
DiffuseTexture::DiffuseTexture() { name = DIFFUSE; }

DiffuseTexture::DiffuseTexture(TextureData *data) : Texture(data) {
  name = DIFFUSE;
  if (!data)
    set_dummy_TextureData(this);
}

void DiffuseTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void DiffuseTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void DiffuseTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + DIFFUSE);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void DiffuseTexture::set(TextureData *texture) {
  clean();
  width = texture->width;
  height = texture->height;
  data.resize(width * height);
  data_format = static_cast<Texture::FORMAT>(texture->data_format);
  internal_format = static_cast<Texture::FORMAT>(texture->internal_format);
  data_type = static_cast<Texture::FORMAT>(texture->data_type);
  has_transparency = false;
  for (unsigned int i = 0; i < width * height; i++) {
    data[i] = texture->data[i];
    if ((data[i] & 0xFF000000) != 0xFF000000)  // check for transparency
      has_transparency = true;
  }
}

const char *DiffuseTexture::getTextureTypeCStr() { return texture_type_c_str[DIFFUSE]; }

/****************************************************************************************************************************/
NormalTexture::NormalTexture() { name = NORMAL; }

NormalTexture::NormalTexture(TextureData *texture) : Texture(texture) {
  name = NORMAL;
  if (data.empty())
    set_dummy_TextureData_normals(this);
}

void NormalTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + NORMAL);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void NormalTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + NORMAL);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void NormalTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + NORMAL);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *NormalTexture::getTextureTypeCStr() { return texture_type_c_str[NORMAL]; }

/****************************************************************************************************************************/
MetallicTexture::MetallicTexture() { name = METALLIC; }

MetallicTexture::MetallicTexture(TextureData *data) : Texture(data) {
  name = METALLIC;
  if (!data)
    set_dummy_TextureData(this);
}

void MetallicTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + METALLIC);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void MetallicTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + METALLIC);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void MetallicTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + METALLIC);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *MetallicTexture::getTextureTypeCStr() { return texture_type_c_str[METALLIC]; }

/****************************************************************************************************************************/
RoughnessTexture::RoughnessTexture() { name = ROUGHNESS; }

RoughnessTexture::RoughnessTexture(TextureData *data) : Texture(data) {
  name = ROUGHNESS;
  if (!data)
    set_dummy_TextureData(this);
}

void RoughnessTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + ROUGHNESS);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void RoughnessTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + ROUGHNESS);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void RoughnessTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + ROUGHNESS);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *RoughnessTexture::getTextureTypeCStr() { return texture_type_c_str[ROUGHNESS]; }

/****************************************************************************************************************************/
AmbiantOcclusionTexture::AmbiantOcclusionTexture() { name = AMBIANTOCCLUSION; }

AmbiantOcclusionTexture::AmbiantOcclusionTexture(TextureData *data) : Texture(data) {
  name = AMBIANTOCCLUSION;
  if (!data)
    set_dummy_TextureData(this);
}

void AmbiantOcclusionTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void AmbiantOcclusionTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void AmbiantOcclusionTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *AmbiantOcclusionTexture::getTextureTypeCStr() { return texture_type_c_str[AMBIANTOCCLUSION]; }

/****************************************************************************************************************************/
SpecularTexture::SpecularTexture() { name = SPECULAR; }

SpecularTexture::SpecularTexture(TextureData *data) : Texture(data) {
  name = SPECULAR;
  if (!data)
    set_dummy_TextureData(this);
}

void SpecularTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + SPECULAR);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void SpecularTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + SPECULAR);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void SpecularTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + SPECULAR);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *SpecularTexture::getTextureTypeCStr() { return texture_type_c_str[SPECULAR]; }

/****************************************************************************************************************************/
EmissiveTexture::EmissiveTexture() { name = EMISSIVE; }

EmissiveTexture::EmissiveTexture(TextureData *data) : Texture(data) {
  name = EMISSIVE;
  if (!data)
    set_dummy_TextureData(this);
}

void EmissiveTexture::initializeTexture2D() {
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, data.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  errorCheck(__FILE__, __LINE__);
}

void EmissiveTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void EmissiveTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void EmissiveTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + EMISSIVE);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *EmissiveTexture::getTextureTypeCStr() { return texture_type_c_str[EMISSIVE]; }

/****************************************************************************************************************************/
OpacityTexture::OpacityTexture() { name = OPACITY; }

OpacityTexture::OpacityTexture(TextureData *data) : Texture(data) {
  name = OPACITY;
  if (!data)
    set_dummy_TextureData(this);
}

void OpacityTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + OPACITY);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void OpacityTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + OPACITY);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void OpacityTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + OPACITY);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *OpacityTexture::getTextureTypeCStr() { return texture_type_c_str[OPACITY]; }

/****************************************************************************************************************************/

Generic2DTexture::Generic2DTexture() { name = GENERIC; }

Generic2DTexture::Generic2DTexture(TextureData *data) : Texture(data) { name = GENERIC; }

void Generic2DTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!data.empty())
    Texture::initializeTexture2D();
  shader->setTextureUniforms(location_name, texture_unit);
}

void Generic2DTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void Generic2DTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *Generic2DTexture::getTextureTypeCStr() { return location_name.c_str(); }

void Generic2DTexture::setTextureUnit(unsigned int tex_unit) { texture_unit = tex_unit; }

void Generic2DTexture::setLocationName(std::string _name) { location_name = _name; }

/****************************************************************************************************************************/
CubemapTexture::CubemapTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : Texture() {  //! Move arguments to Texture()
  name = CUBEMAP;
  internal_format = _internal_format;
  data_format = _data_format;
  data_type = _data_type;
  width = _width;
  height = _height;
  mipmaps = 0;
}

void CubemapTexture::setCubeMapTextureData(TextureData *texture) {
  clean();
  width = texture->width;
  height = texture->height;
  internal_format = static_cast<Texture::FORMAT>(texture->internal_format);
  data_format = static_cast<Texture::FORMAT>(texture->data_format);
  data_type = static_cast<Texture::FORMAT>(texture->data_type);
  mipmaps = texture->mipmaps;
  /* In case the raw data is in RGB-RGBA with 8 bits/channel*/
  if (!texture->data.empty()) {
    data.resize(width * height * 6);
    for (unsigned int i = 0; i < width * height * 6; i++)
      data[i] = texture->data[i];
  }
  /* In case raw data is 4 bytes float / channel */
  if (!texture->f_data.empty()) {
    f_data.resize(width * height * 6 * texture->nb_components);
    for (unsigned i = 0; i < width * height * 6 * texture->nb_components; i++)
      f_data[i] = texture->f_data[i];
  }
}

CubemapTexture::CubemapTexture(TextureData *data) : CubemapTexture() {
  name = CUBEMAP;
  if (data)
    setCubeMapTextureData(data);
}

void CubemapTexture::initializeTexture2D() {
  if (!data.empty())
    for (unsigned int i = 1; i <= 6; i++) {
      uint32_t *pointer_to_data = data.data() + (i - 1) * width * height;
      glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (i - 1), 0, internal_format, width, height, 0, data_format, data_type, pointer_to_data);
      errorCheck(__FILE__, __LINE__);
    }
  else
    for (unsigned i = 0; i < 6; i++) {
      glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internal_format, width, height, 0, data_format, data_type, nullptr);
      errorCheck(__FILE__, __LINE__);
    }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void CubemapTexture::generateMipmap() {
  bindTexture();
  glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  unbindTexture();
}

void CubemapTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  bindTexture();
  initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
  unbindTexture();
}

void CubemapTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + name);
  glBindTexture(GL_TEXTURE_CUBE_MAP, sampler2D);
}

void CubemapTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + name);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void CubemapTexture::setNewSize(unsigned w, unsigned h) {
  //! Implements this
}

const char *CubemapTexture::getTextureTypeCStr() { return texture_type_c_str[CUBEMAP]; }

/****************************************************************************************************************************/

GenericCubemapTexture::GenericCubemapTexture(FORMAT internal_format, FORMAT data_format, FORMAT data_type, unsigned width, unsigned height)
    : CubemapTexture(internal_format, data_format, data_type, width, height) {
  name = GENERIC_CUBE;
}

GenericCubemapTexture::GenericCubemapTexture(TextureData *data) : CubemapTexture(data) { name = GENERIC_CUBE; }

void GenericCubemapTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_CUBE_MAP, sampler2D);
}

void GenericCubemapTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + texture_unit);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void GenericCubemapTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  bindTexture();
  initializeTexture2D();
  shader->setTextureUniforms(location_name, texture_unit);
  unbindTexture();
}

const char *GenericCubemapTexture::getTextureTypeCStr() { return location_name.c_str(); }

/****************************************************************************************************************************/
IrradianceTexture::IrradianceTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : CubemapTexture(_internal_format, _data_format, _data_type, _width, _height) {
  name = IRRADIANCE;
}

IrradianceTexture::IrradianceTexture(TextureData *data) : IrradianceTexture() {
  name = IRRADIANCE;
  if (data)
    setCubeMapTextureData(data);
}

const char *IrradianceTexture::getTextureTypeCStr() { return texture_type_c_str[IRRADIANCE]; }

/****************************************************************************************************************************/
/* This is loaded from the disk as an .hdr image , with 4 bytes float for texel format on each channel...
 * note: make it generic so that we can generate it on the fly ?
 */

EnvironmentMap2DTexture::EnvironmentMap2DTexture(FORMAT _internal_format, FORMAT _data_format, FORMAT _data_type, unsigned _width, unsigned _height)
    : Texture() {
  name = ENVMAP2D;
  internal_format = _internal_format;
  data_format = _data_format;
  data_type = _data_type;
  width = _width;
  height = _height;
}

EnvironmentMap2DTexture::EnvironmentMap2DTexture(TextureData *data) : Texture(data) { name = ENVMAP2D; }

void EnvironmentMap2DTexture::initializeTexture2D() {
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, f_data.data());
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glGenerateMipmap(GL_TEXTURE_2D);
  errorCheck(__FILE__, __LINE__);
}

void EnvironmentMap2DTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + ENVMAP2D);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  if (!f_data.empty())
    EnvironmentMap2DTexture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void EnvironmentMap2DTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + ENVMAP2D);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void EnvironmentMap2DTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + ENVMAP2D);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *EnvironmentMap2DTexture::getTextureTypeCStr() { return texture_type_c_str[ENVMAP2D]; }

/****************************************************************************************************************************/

FrameBufferTexture::FrameBufferTexture() : Texture() {
  name = FRAMEBUFFER;
  internal_format = RGBA16F;
  data_format = BGRA;
  data_type = UBYTE;
}

FrameBufferTexture::FrameBufferTexture(TextureData *_data) : FrameBufferTexture() {
  if (_data != nullptr) {
    width = _data->width;
    height = _data->height;
    internal_format = static_cast<Texture::FORMAT>(_data->internal_format);
    data_format = static_cast<Texture::FORMAT>(_data->data_format);
    data_type = static_cast<Texture::FORMAT>(_data->data_type);
    this->data.clear();
  }
}

FrameBufferTexture::FrameBufferTexture(unsigned _width, unsigned _height) : FrameBufferTexture() {
  width = _width;
  height = _height;
}

void FrameBufferTexture::initializeTexture2D() {
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void FrameBufferTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
  FrameBufferTexture::initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void FrameBufferTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void FrameBufferTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER);
  glBindTexture(GL_TEXTURE_2D, 0);
}

const char *FrameBufferTexture::getTextureTypeCStr() { return texture_type_c_str[FRAMEBUFFER]; }

/****************************************************************************************************************************/

BRDFLookupTexture::BRDFLookupTexture() : Texture() { name = BRDFLUT; }

BRDFLookupTexture::BRDFLookupTexture(TextureData *data) : Texture(data) { name = BRDFLUT; }

void BRDFLookupTexture::bindTexture() {
  glActiveTexture(GL_TEXTURE0 + name);
  glBindTexture(GL_TEXTURE_2D, sampler2D);
}

void BRDFLookupTexture::unbindTexture() {
  glActiveTexture(GL_TEXTURE0 + name);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void BRDFLookupTexture::setGlData(Shader *shader) {
  glGenTextures(1, &sampler2D);
  bindTexture();
  initializeTexture2D();
  shader->setTextureUniforms(texture_type_c_str[name], name);
}

void BRDFLookupTexture::initializeTexture2D() {
  if (!data.empty()) {
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, f_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
}

const char *BRDFLookupTexture::getTextureTypeCStr() { return texture_type_c_str[BRDFLUT]; }
