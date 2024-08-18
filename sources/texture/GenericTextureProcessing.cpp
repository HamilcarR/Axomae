#include "GenericTextureProcessing.h"
#include "device/opengl/init_3D.h"

TextureData::TextureData()
    : width(0), height(0), nb_components(1), mipmaps(5), internal_format(GL_RGBA), data_format(GL_BGRA), data_type(GL_UNSIGNED_BYTE) {}
TextureData::TextureData(const TextureData &copy)
    : width(copy.width),
      height(copy.height),
      name(copy.name),
      nb_components(copy.nb_components),
      mipmaps(copy.mipmaps),
      internal_format(copy.internal_format),
      data_format(copy.data_format),
      data_type(copy.data_type) {
  if (!copy.data.empty())
    data = copy.data;
  if (!copy.f_data.empty())
    f_data = copy.f_data;
}

TextureData::TextureData(TextureData &&move) noexcept
    : width(move.width),
      height(move.height),
      nb_components(move.nb_components),
      mipmaps(move.mipmaps),
      internal_format(move.internal_format),
      data_format(move.data_format),
      data_type(move.data_type) {

  if (!move.data.empty())
    data = std::move(move.data);
  if (!move.f_data.empty())
    f_data = std::move(move.f_data);
  name = move.name;
}

TextureData &TextureData::operator=(const TextureData &from) {
  if (this != &from) {
    width = from.width;
    height = from.height;
    data_format = from.data_format;
    data_type = from.data_type;
    internal_format = from.internal_format;
    mipmaps = from.mipmaps;
    nb_components = from.nb_components;
    if (!from.data.empty())
      data = from.data;
    if (!from.f_data.empty())
      f_data = from.f_data;

    name = from.name;
  }
  return *this;
}

TextureData &TextureData::operator=(TextureData &&from) noexcept {
  if (this != &from) {
    width = from.width;
    height = from.height;
    data_format = from.data_format;
    data_type = from.data_type;
    internal_format = from.internal_format;
    mipmaps = from.mipmaps;
    nb_components = from.nb_components;
    if (!from.data.empty())
      data = std::move(from.data);
    if (!from.f_data.empty())
      f_data = std::move(from.f_data);
    name = from.name;
  }
  return *this;
}

void TextureData::clean() {
  f_data.clear();
  data.clear();
  width = 0;
  height = 0;
  name = "";
}