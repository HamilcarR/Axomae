#ifndef GENERICTEXTUREPROCESSING_H
#define GENERICTEXTUREPROCESSING_H
#include "internal/memory/Serializable.h"
#include <GL/glew.h>
#include <cstdint>
#include <internal/macro/project_macros.h>
#include <string>
#include <vector>

/**
 * @brief Class for raw binary data of textures
 * !Note : While using HDR envmap , the data format is still uint32_t , as we wont need to use any other texture format
 * than .hdr files
 *
 */

template<class T>
class TextureData : public core::memory::Serializable {
 public:
  enum CHANNELS : unsigned { RGB = 0, RGBA = 1 };
  uint8_t nb_components{};
  uint32_t width{};
  uint32_t height{};
  uint32_t mipmaps{};
  uint32_t internal_format{};
  uint32_t data_format{};
  uint32_t data_type{};
  std::string name;
  std::vector<T> data;

 public:
  TextureData();
  ~TextureData() override = default;
  TextureData(const TextureData &copy) = default;
  TextureData(TextureData &&move) noexcept = default;
  TextureData &operator=(const TextureData &copy) = default;
  TextureData &operator=(TextureData &&move) noexcept = default;
  void clean();
  ax_no_discard std::vector<uint8_t> serialize() const override;
};

using F32TexData = TextureData<float>;
using U32TexData = TextureData<uint32_t>;

template<class T>
TextureData<T>::TextureData() : nb_components(1), mipmaps(5), internal_format(GL_RGBA), data_format(GL_BGRA), data_type(GL_UNSIGNED_BYTE) {}

template<class T>
void TextureData<T>::clean() {
  data.clear();
  width = 0;
  height = 0;
  name = "";
}

template<class T>
std::vector<uint8_t> TextureData<T>::serialize() const {
  std::vector<uint8_t> serialized;
  /*  std::size_t this_size = 0;
    this_size += sizeof(nb_components);
    this_size += sizeof(width);
    this_size += sizeof(height);
    this_size += sizeof(mipmaps);
    this_size += sizeof(internal_format);
    this_size += sizeof(data_format);
    this_size += sizeof(data_type);
    this_size += name.size();
    this_size += data.size() * sizeof(T);
    serialized.reserve(this_size);*/
  return serialized;
}

#endif