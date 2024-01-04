#ifndef IMAGE_H
#define IMAGE_H
#include "Thumbnail.h"
#include <QPixmap>
#include <string>
namespace image {
  struct Metadata {
    std::string name{};
    std::string format{};
    unsigned int height{};
    unsigned int width{};
    unsigned int channels{};
    bool color_corrected{};
  };

  template<class TYPE>
  class RawImageHolder {
   public:
    RawImageHolder() = default;
    /*Will perform a move on assign*/
    RawImageHolder(const std::vector<TYPE> &assign, image::Metadata _metadata_) : metadata_(std::move(_metadata_)), data(assign) {
      icon = Thumbnail<TYPE>(data, metadata_.width, metadata_.height, metadata_.channels, !metadata_.color_corrected);
    }
    const QPixmap &thumbnail() { return icon.getIcon(); }
    Metadata metadata() { return metadata_; }
    std::string name() { return metadata_.name; }

   public:
    Metadata metadata_;
    std::vector<TYPE> data;
    Thumbnail<TYPE> icon;
  };
}  // namespace image

#endif