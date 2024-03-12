#ifndef GENERICTEXTUREPROCESSING_H
#define GENERICTEXTUREPROCESSING_H

#include "IAxObject.h"
#include "constants.h"

/**
 * @brief Class for raw binary data of textures
 * !Note : While using HDR envmap , the data format is still uint32_t , as we wont need to use any other texture format
 * than .hdr files
 *
 */
class TextureData {  // TODO : Make this inherit RawImageHolder
 public:
  enum CHANNELS : unsigned { RGB = 0, RGBA = 1 };

  TextureData();

  ~TextureData() = default;

  TextureData(const TextureData &copy);

  TextureData(TextureData &&move) noexcept;

  /**
   * @brief Copy a texture
   *
   * Provides deep copy of the object , but doesn't do the cleanup for the copied object
   *
   * @param from The texture to be copied
   * @return * TextureData& Deep copy of the original TextureData object
   */
  TextureData &operator=(const TextureData &from);

  TextureData &operator=(TextureData &&from) noexcept;

  void clean();

 public:
  unsigned int width;         /**<Width of the texture*/
  unsigned int height;        /**<Height of the texture*/
  std::string name;           /**<Name of the texture*/
  std::vector<uint32_t> data; /*<1D array raw data of the texture*/
  std::vector<float> f_data;  /*<1D float array raw data of the texture (HDR)*/
  unsigned nb_components;     /*<Number of channels*/
  unsigned mipmaps;           /*<Number of mipmaps*/
  unsigned internal_format;
  unsigned data_format;
  unsigned data_type;
};

/******************************************************************************************************************************************************************************************************************/
class GenericTextureProcessing : public IAxObject {
 public:
  [[nodiscard]] virtual bool isDimPowerOfTwo(int dim) const = 0;
  [[nodiscard]] virtual bool isValidDim(int dim) const = 0;
};

#endif