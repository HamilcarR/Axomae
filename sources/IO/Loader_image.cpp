#include "Image.h"
#include "ImageDatabase.h"
#include "Loader.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "GenericException.h"
#include "ImfChannelList.h"
#include "Logger.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "string/axomae_str_utils.h"
#include <ImfArray.h>
#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <fstream>
namespace exception {
  const char *INVALID_PATH_ENVMAP = "Must be an HDR/EXR format .";

  class LoadImagePathException : public GenericException {
   public:
    explicit LoadImagePathException(const std::string &path) : GenericException() {
      GenericException::saveErrorString(std::string("Failed processing path for image : ") + path);
    }
  };

  class LoadImageDimException : public GenericException {
   public:
    explicit LoadImageDimException(int width, int height) : GenericException() {
      std::string dim = std::string("width : ") + std::to_string(width) + std::string(" height:") + std::to_string(height);
      GenericException::saveErrorString(std::string("Image dimensions error: ") + dim);
    }
  };

  class LoadImageChannelException : public GenericException {
   public:
    explicit LoadImageChannelException(int channels) : GenericException() {
      std::string chan = std::string("channels number : ") + std::to_string(channels);
      GenericException::saveErrorString(std::string("Image channel error: ") + chan);
    }
  };

  class InvalidImageFormat : public GenericException {
   public:
    explicit InvalidImageFormat(const std::string &error) : GenericException() {
      GenericException::saveErrorString(std::string("Image format error: ") + error);
    }
  };
}  // namespace exception

namespace IO {

  void Loader::writeHdr(const char *path, const image::ImageHolder<float> &image, bool flip) {
    std::string final_path = path;
    final_path += ".hdr";
    stbi_flip_vertically_on_write(flip);
    int n = stbi_write_hdr(final_path.c_str(),
                           static_cast<int>(image.metadata.width),
                           static_cast<int>(image.metadata.height),
                           static_cast<int>(image.metadata.channels),
                           image.data.data());
    if (n == 0) {
      throw exception::LoadImagePathException(path);
    }
  }

  static bool check_exr(const char f[32]) { return f[0] == 0x76 && f[1] == 0x2f && f[2] == 0x31 && f[3] == 0x01; }
  static bool check_radiance(const char f[32]) { return std::strstr(f, "#?RADIANCE") != nullptr; }

  static void load_magic_header(const char *filename, char r_format[32]) {
    std::ifstream input_str(filename, std::ios::binary);
    if (!input_str.is_open())
      throw exception::LoadImagePathException(filename);
    input_str.read(r_format, 31 * sizeof(char));
    if (!check_exr(r_format) && !check_radiance(r_format)) {
      input_str.close();
      throw exception::InvalidImageFormat(exception::INVALID_PATH_ENVMAP);
    }
    input_str.close();
  }

  image::ImageHolder<float> Loader::loadHdrEnvmap(const char *path, bool store) {
    try {
      char format[32] = {0};
      load_magic_header(path, format);
      if (check_exr(format))
        return loadExrFile(path, store);
      if (check_radiance(format))
        return loadRadianceFile(path, store);
      throw exception::InvalidImageFormat(exception::INVALID_PATH_ENVMAP);
    } catch (exception::GenericException &e) {
      throw;
    }
  }
  /***************************************************************************************/
  /* Exr file format*/
  image::ImageHolder<float> Loader::loadExrFile(const char *path, bool store) {
    std::string str_path(path);
    std::string filename = utils::string::tokenize(str_path, '/').back();

    try {
      image::ImageHolder<float> img;
      Imf::InputFile input_file(path);
      Imath::Box2i win = input_file.header().dataWindow();
      bool alpha = input_file.header().channels().findChannel("A");
      img.metadata.width = win.max.x - win.min.x + 1;
      img.metadata.height = win.max.y - win.min.y + 1;
      img.metadata.name = filename;
      img.metadata.color_corrected = false;
      img.metadata.is_hdr = true;
      img.metadata.channels = alpha ? 4 : 3;
      img.data.resize(img.metadata.width * img.metadata.height * img.metadata.channels);
      Imf::FrameBuffer framebuffer;
      std::vector<float> pixels;

      pixels.resize(img.metadata.width * img.metadata.height * img.metadata.channels);
      framebuffer.insert("R",
                         Imf::Slice(Imf::FLOAT,
                                    (char *)(pixels.data()),
                                    sizeof(float) * img.metadata.channels,
                                    sizeof(float) * img.metadata.channels * img.metadata.width));
      framebuffer.insert("G",
                         Imf::Slice(Imf::FLOAT,
                                    (char *)(pixels.data() + 1),
                                    sizeof(float) * img.metadata.channels,
                                    sizeof(float) * img.metadata.channels * img.metadata.width));
      framebuffer.insert("B",
                         Imf::Slice(Imf::FLOAT,
                                    (char *)(pixels.data() + 2),
                                    sizeof(float) * img.metadata.channels,
                                    sizeof(float) * img.metadata.channels * img.metadata.width));
      if (alpha) {
        framebuffer.insert("A",
                           Imf::Slice(Imf::FLOAT,
                                      (char *)pixels.data() + 3,
                                      sizeof(float) * img.metadata.channels,
                                      sizeof(float) * img.metadata.channels * img.metadata.width));
      }
      input_file.setFrameBuffer(framebuffer);
      input_file.readPixels(win.min.y, win.max.y);
      std::memcpy(img.data.data(), pixels.data(), img.metadata.channels * img.metadata.width * img.metadata.height * sizeof(float));

      if (store) {
        HdrImageDatabase *hdr_database = resource_database->getHdrDatabase();
        database::image::store<float>(*hdr_database, false, img.data, img.metadata);
      }
      return img;
    } catch (Iex::BaseExc &e) {
      throw exception::InvalidImageFormat(e.what());
    }
  }

  /***************************************************************************************/
  /* Radiance file format*/
  image::ImageHolder<float> Loader::loadRadianceFile(const char *path, bool store) {
    controller::ProgressManagerHelper helper(this);
    int width = -1, height = -1, channels = -1;
    float *data = stbi_loadf(path, &width, &height, &channels, 0);
    initProgress("Importing environment map", (float)(width * height * channels));
    helper.notifyProgress(controller::ProgressManagerHelper::ONE_FOURTH);
    if (!data) {
      helper.reset();
      throw exception::LoadImagePathException(path);
    }
    AX_ASSERT(width > 0 && height > 0, "Wrong size \"width\" or \"height\" for imported HDR image.");
    AX_ASSERT(channels == 1 || channels == 2 || channels == 3 || channels == 4, "Channel number invalid for imported HDR image.");

    std::vector<float> image_data{};
    image_data.reserve(width * height * channels);
    for (int i = 0; i < width * height * channels; i++)
      image_data.push_back(data[i]);

    std::string path_str(path);
    std::string name = utils::string::tokenize(path_str, '/').back();

    image::Metadata metadata;
    metadata.channels = channels;
    metadata.width = width;
    metadata.height = height;
    metadata.is_hdr = true;
    metadata.name = name;
    metadata.color_corrected = false;

    if (store) {
      HdrImageDatabase *hdr_database = resource_database->getHdrDatabase();
      database::image::store<float>(*hdr_database, false, image_data, metadata);
    }
    stbi_image_free(data);
    helper.reset();
    return {image_data, metadata};
  }

}  // namespace IO