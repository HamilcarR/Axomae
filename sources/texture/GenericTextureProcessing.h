#ifndef GENERICTEXTUREPROCESSING_H
#define GENERICTEXTUREPROCESSING_H
#include <GL/glew.h>
#include <cstdint>
#include <string>
#include <vector>
/**
 * @brief Class for raw binary data of textures
 * !Note : While using HDR envmap , the data format is still uint32_t , as we wont need to use any other texture format
 * than .hdr files
 *
 */

template<class T>
class TextureData {
 public:
  enum CHANNELS : unsigned { RGB = 0, RGBA = 1 };

 public:
  unsigned int width;
  unsigned int height;
  std::string name;
  std::vector<T> data;
  unsigned nb_components;
  unsigned mipmaps;
  unsigned internal_format;
  unsigned data_format;
  unsigned data_type;

 public:
  TextureData();
  ~TextureData() = default;
  TextureData(const TextureData &copy) = default;
  TextureData(TextureData &&move) noexcept = default;
  TextureData &operator=(const TextureData &copy) = default;
  TextureData &operator=(TextureData &&move) noexcept = default;
  void clean();
};

using F32TexData = TextureData<float>;
using U32TexData = TextureData<uint32_t>;

template<class T>
TextureData<T>::TextureData()
    : width(0), height(0), nb_components(1), mipmaps(5), internal_format(GL_RGBA), data_format(GL_BGRA), data_type(GL_UNSIGNED_BYTE) {}

template<class T>
void TextureData<T>::clean() {
  data.clear();
  width = 0;
  height = 0;
  name = "";
}

/******************************************************************************************************************************************************************************************************************/

#endif