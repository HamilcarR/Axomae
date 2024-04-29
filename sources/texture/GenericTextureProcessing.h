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

 public:
  unsigned int width;
  unsigned int height;
  std::string name;
  std::vector<uint32_t> data;
  std::vector<float> f_data;
  unsigned nb_components;
  unsigned mipmaps;
  unsigned internal_format;
  unsigned data_format;
  unsigned data_type;

 public:
  TextureData();
  ~TextureData() = default;
  TextureData(const TextureData &copy);
  TextureData(TextureData &&move) noexcept;
  TextureData &operator=(const TextureData &copy);
  TextureData &operator=(TextureData &&move) noexcept;
  void clean();
};

/******************************************************************************************************************************************************************************************************************/

#endif