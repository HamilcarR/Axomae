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
    void flip_v();
    void flip_u();
    void clear() { data.clear(); }
  };

  template<class T>
  ImageHolder<T>::ImageHolder(const std::vector<T> &img, const image::Metadata &meta) : data(img), metadata(meta) {}

  template<class T>
  void ImageHolder<T>::flip_v() {
    for (int y = 0; y < metadata.height / 2; y++) {
      for (int x = 0; x < metadata.width; x++) {
        int cur = (y * metadata.width + x) * metadata.channels;
        int inv = ((metadata.height - 1 - y) * metadata.width + x) * metadata.channels;
        for (int k = 0; k < metadata.channels; k++)
          std::swap(data[cur + k], data[inv + k]);
      }
    }
  }
  template<class T>
  void ImageHolder<T>::flip_u() {
    for (int y = 0; y < metadata.height; y++) {
      for (int x = 0; x < metadata.width / 2; x++) {
        int cur = (y * metadata.width + x) * metadata.channels;
        int inv = (y * metadata.width + (metadata.width - 1 - x)) * metadata.channels;
        for (int k = 0; k < metadata.channels; k++)
          std::swap(data[cur + k], data[inv + k]);
      }
    }
  }
  /*******************************************************************************************************************************************/

  template<class TYPE>
  class ThumbnailImageHolder : public ImageHolder<TYPE> {
    using BASETYPE = ImageHolder<TYPE>;

   public:
    Thumbnail icon;

   public:
    ThumbnailImageHolder() = default;
    ThumbnailImageHolder(const std::vector<TYPE> &assign,
                         const image::Metadata &_metadata_,
                         controller::ProgressStatus *status,
                         bool generate_thumbnail = true);

    const QPixmap &thumbnail();
    std::string name();
  };

  template<class T>
  ThumbnailImageHolder<T>::ThumbnailImageHolder(const std::vector<T> &assign,
                                                const image::Metadata &_metadata_,
                                                controller::ProgressStatus *status,
                                                bool generate_thumbnail)
      : ImageHolder<T>(assign, _metadata_) {
    if (generate_thumbnail)
      icon = Thumbnail(BASETYPE::data, _metadata_, status);
  }

  template<class T>
  const QPixmap &ThumbnailImageHolder<T>::thumbnail() {
    return icon.getIcon();
  }

  template<class T>
  std::string ThumbnailImageHolder<T>::name() {
    return BASETYPE::metadata.name;
  }

  /*******************************************************************************************************************************************/

}  // namespace image

#endif