#ifndef GENERICTEXTUREPROCESSING_H
#define GENERICTEXTUREPROCESSING_H

#include "constants.h"
#include "init_3D.h"
/**
 * @brief Class for raw binary data of textures
 * !Note : While using HDR envmap , the data format is still uint32_t , as we wont need to use any other texture format
 * than .hdr files
 *
 */
class TextureData {  // TODO : Make this inherit RawImageHolder
 public:
  /**
   * @brief Rgb channels types
   *
   */
  enum CHANNELS : unsigned { RGB = 0, RGBA = 1 };

  /**
   * @brief Construct a new Texture Data object
   *
   */
  TextureData() {
    width = 0;
    height = 0;
    nb_components = 1;
    mipmaps = 5;
    internal_format = GL_RGBA;
    data_format = GL_BGRA;
    data_type = GL_UNSIGNED_BYTE;
  }

  /**
   * @brief Copy a texture
   *
   * Provides deep copy of the object , but doesn't do the cleanup for the copied object
   *
   * @param from The texture to be copied
   * @return * TextureData& Deep copy of the original TextureData object
   */
  TextureData &operator=(const TextureData &from) {
    if (this != &from) {
      width = from.width;
      height = from.height;
      data_format = from.data_format;
      data_type = from.data_type;
      internal_format = from.internal_format;
      mipmaps = from.mipmaps;
      nb_components = from.nb_components;
      if (!from.data.empty()) {
        data = from.data;
      }
      if (!from.f_data.empty()) {
        f_data = from.f_data;
      }
      name = from.name;
    }
    return *this;
  }

  /**
   * @brief Free the object
   *
   */
  void clean() {
    f_data.clear();
    data.clear();
    width = 0;
    height = 0;
    name = "";
  }

 public:
  unsigned int width;         /**<Width of the texture*/
  unsigned int height;        /**<Height of the texture*/
  std::string name;           /**<Name of the texture*/
  std::vector<uint32_t> data; /*<1D array raw data of the texture*/
  std::vector<float> f_data;  /*<1D float array raw data of the texture (HDR)*/
  unsigned nb_components;     /*<Number of channels*/
  unsigned mipmaps;           /*<Number of mipmaps*/
  GLenum internal_format;
  GLenum data_format;
  GLenum data_type;
};

/******************************************************************************************************************************************************************************************************************/
class GenericTextureProcessing {
 public:
  virtual bool isDimPowerOfTwo(int dim) const = 0;
  virtual bool isValidDim(int dim) const = 0;
};

#endif