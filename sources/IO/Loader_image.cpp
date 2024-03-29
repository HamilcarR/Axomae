#include "Image.h"
#include "ImageDatabase.h"
#include "Loader.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "GenericException.h"
#include "axomae_utils.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace exception {
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
}  // namespace exception

namespace IO {

  image::ImageHolder<float> Loader::loadHdr(const char *path, bool store) {
    controller::ProgressManagerHelper helper(this);
    helper.notifyProgress(controller::ProgressManagerHelper::ZERO);
    int width = -1, height = -1, channels = -1;
    float *data = stbi_loadf(path, &width, &height, &channels, 0);
    initProgress("Importing environment map", width * height * channels);
    if (!data)
      throw exception::LoadImagePathException(path);
    AX_ASSERT(width > 0 && height > 0);
    AX_ASSERT(channels == 1 || channels == 2 || channels == 3 || channels == 4);

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
    helper.notifyProgress(controller::ProgressManagerHelper::COMPLETE);
    return image::ImageHolder<float>(image_data, metadata);
  }

  void Loader::writeHdr(const char *path, const image::ImageHolder<float> &image) {
    int n = stbi_write_hdr(path,
                           static_cast<int>(image.metadata.width),
                           static_cast<int>(image.metadata.height),
                           static_cast<int>(image.metadata.channels),
                           image.data.data());
    if (n == 0) {
      throw exception::LoadImagePathException(path);
    }
  }

}  // namespace IO