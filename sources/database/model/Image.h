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
  class ImageHolder {
   public:
    explicit ImageHolder(const std::vector<TYPE> &img, image::Metadata &meta) : data(img), metadata(meta) {}

   public:
    std::vector<TYPE> data;
    Metadata metadata;
  };

  template<class TYPE>
  class ThumbnailImageHolder : public ImageHolder<TYPE> {
    using BASETYPE = ImageHolder<TYPE>;

   public:
    ThumbnailImageHolder() = default;
    /*Will perform a move on assign*/
    ThumbnailImageHolder(const std::vector<TYPE> &assign, image::Metadata _metadata_, controller::ProgressStatus *status)
        : ImageHolder<TYPE>(assign, _metadata_) {
      icon = Thumbnail<TYPE>(BASETYPE::data,
                             BASETYPE::metadata.width,
                             BASETYPE::metadata.height,
                             BASETYPE::metadata.channels,
                             !BASETYPE::metadata.color_corrected,
                             status);
    }
    const QPixmap &thumbnail() { return icon.getIcon(); }
    Metadata metadata() { return BASETYPE::metadata; }
    std::string name() { return BASETYPE::metadata.name; }

   public:
    Thumbnail<TYPE> icon;
  };
}  // namespace image

#endif