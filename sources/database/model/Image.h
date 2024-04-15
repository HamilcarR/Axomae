#ifndef IMAGE_H
#define IMAGE_H
#include "Metadata.h"
#include "Thumbnail.h"
#include <QPixmap>
#include <string>
namespace image {

  template<class TYPE>
  class ImageHolder {
   public:
    std::vector<TYPE> data{};
    Metadata metadata{};

   public:
    ImageHolder() = default;
    ImageHolder(const std::vector<TYPE> &img, const image::Metadata &meta);
    virtual ~ImageHolder() = default;
    ImageHolder(const ImageHolder<TYPE> &copy) = default;
    ImageHolder(ImageHolder<TYPE> &&assign) noexcept = default;
    ImageHolder<TYPE> &operator=(const ImageHolder<TYPE> &copy) = default;
    ImageHolder<TYPE> &operator=(ImageHolder<TYPE> &&assign) noexcept = default;
  };

  template<class T>
  ImageHolder<T>::ImageHolder(const std::vector<T> &img, const image::Metadata &meta) : data(img), metadata(meta) {}

  /*******************************************************************************************************************************************/

  template<class TYPE>
  class ThumbnailImageHolder : public ImageHolder<TYPE> {
    using BASETYPE = ImageHolder<TYPE>;

   public:
    Thumbnail icon;

   public:
    ThumbnailImageHolder() = default;
    ThumbnailImageHolder(const std::vector<TYPE> &assign, const image::Metadata &_metadata_, controller::ProgressStatus *status);

    const QPixmap &thumbnail();
    Metadata metadata();
    std::string name();
  };

  template<class T>
  ThumbnailImageHolder<T>::ThumbnailImageHolder(const std::vector<T> &assign, const image::Metadata &_metadata_, controller::ProgressStatus *status)
      : ImageHolder<T>(assign, _metadata_) {
    icon = Thumbnail(BASETYPE::data, _metadata_, status);
  }

  template<class T>
  const QPixmap &ThumbnailImageHolder<T>::thumbnail() {
    return icon.getIcon();
  }

  template<class T>
  Metadata ThumbnailImageHolder<T>::metadata() {
    return BASETYPE::metadata;
  }

  template<class T>
  std::string ThumbnailImageHolder<T>::name() {
    return BASETYPE::metadata.name;
  }

  /*******************************************************************************************************************************************/

}  // namespace image

#endif