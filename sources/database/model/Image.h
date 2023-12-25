#ifndef IMAGE_H
#define IMAGE_H
#include "Thumbnail.h"
#include <QPixmap>
#include <string>
namespace image {
  struct Metadata {
    std::string name;
    std::string format;
    unsigned int height;
    unsigned int width;
    unsigned int channels;
  };

  template<class TYPE>
  class RawImageHolder {
   public:
    RawImageHolder() = default;
    /*Will perform a move on assign*/
    RawImageHolder(const std::vector<TYPE> &assign, image::Metadata _metadata_, int width, int height, int channels)
        : metadata_(std::move(_metadata_)), data(assign) {
      icon = Thumbnail(data, width, height, false, channels);
    }
    QPixmap thumbnail() { return icon.icon; }
    Metadata metadata() { return metadata_; }

   public:
    Metadata metadata_;
    std::vector<TYPE> data;
    Thumbnail icon;
  };
}  // namespace image

#endif