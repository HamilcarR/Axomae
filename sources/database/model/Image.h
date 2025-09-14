#ifndef IMAGE_H
#define IMAGE_H
#include "ImageHolderDataStorageInterface.hpp"
#include "Metadata.h"
#include "Thumbnail.h"
#include <QPixmap>
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/axstd/span.h>
#include <memory>
#include <string>
#include <type_traits>
namespace image {

  template<class T>
  class ImageHolderDataDeviceStorage : public ImageHolderDataStorageInterface<T> {
   private:
    axstd::managed_vector<T> internal_data{};

   public:
    ImageHolderDataDeviceStorage() = default;

    template<class COLLECTION, typename = std::enable_if_t<std::is_convertible_v<typename COLLECTION::value_type, T>>>
    ImageHolderDataDeviceStorage(const COLLECTION &buffer) : internal_data(buffer) {}

    T *data() override { return internal_data.data(); }

    const T *data() const override { return internal_data.data(); }

    std::size_t size() const override { return internal_data.size(); }

    T &operator[](std::size_t index) override { return internal_data[index]; }

    const T &operator[](std::size_t index) const override { return internal_data[index]; }

    void reserve(std::size_t size) override { internal_data.reserve(size); }

    void resize(std::size_t size, const T &value = T()) override { internal_data.resize(size, value); }

    void clear() override { internal_data.clear(); }
  };

  template<class T>
  class ImageHolderDataHostStorage : public ImageHolderDataStorageInterface<T> {
   private:
    std::vector<T> internal_data{};

   public:
    ImageHolderDataHostStorage() = default;

    template<class COLLECTION, typename = std::enable_if_t<std::is_convertible_v<typename COLLECTION::value_type, T>>>
    ImageHolderDataHostStorage(const COLLECTION &buffer) : internal_data(buffer.begin(), buffer.end()) {}

    T *data() override { return internal_data.data(); }

    const T *data() const override { return internal_data.data(); }

    std::size_t size() const override { return internal_data.size(); }

    T &operator[](std::size_t index) override { return internal_data[index]; }

    const T &operator[](std::size_t index) const override { return internal_data[index]; }

    void reserve(std::size_t size) override { internal_data.reserve(size); }

    void resize(std::size_t size, const T &value = T()) override { internal_data.resize(size, value); }

    void clear() override { internal_data.clear(); }
  };

  template<class T>
  class ImageViewDataStorage : public ImageHolderDataStorageInterface<T> {
   private:
    axstd::span<T> internal_data{};

   public:
    ImageViewDataStorage() = default;

    ImageViewDataStorage(axstd::span<T> span) : internal_data(span) {}

    T *data() override { return internal_data.data(); }

    const T *data() const override { return internal_data.data(); }

    std::size_t size() const override { return internal_data.size(); }

    T &operator[](std::size_t index) override { return internal_data[index]; }

    const T &operator[](std::size_t index) const override { return internal_data[index]; }

    void reserve(std::size_t) override {}

    void resize(std::size_t, const T & = T()) override {}

    void clear() override { internal_data = axstd::span<T>(); }
  };

  struct DeviceHolder {};
  struct HostHolder {};
  struct ViewHolder {};

  template<class TYPE>
  class ImageHolder {
   private:
    std::unique_ptr<ImageHolderDataStorageInterface<TYPE>> internal_data;

   public:
    Metadata metadata{};

    ImageHolder() : internal_data(std::make_unique<ImageHolderDataHostStorage<TYPE>>()) {}

    ImageHolder(std::unique_ptr<ImageHolderDataStorageInterface<TYPE>> dep) : internal_data(std::move(dep)) {}

    ImageHolder(const axstd::managed_vector<TYPE> &img, const image::Metadata &meta)
        : internal_data(std::make_unique<ImageHolderDataDeviceStorage<TYPE>>(img)), metadata(meta) {}

    ImageHolder(const std::vector<TYPE> &img, const image::Metadata &meta)
        : internal_data(std::make_unique<ImageHolderDataHostStorage<TYPE>>(img)), metadata(meta) {}

    ImageHolder(const axstd::span<TYPE> &img, const image::Metadata &meta)
        : internal_data(std::make_unique<ImageViewDataStorage<TYPE>>(img)), metadata(meta) {}

    virtual ~ImageHolder() = default;

    ImageHolder(const ImageHolder<TYPE> &copy) = default;

    ImageHolder(ImageHolder<TYPE> &&assign) noexcept = default;

    ImageHolder<TYPE> &operator=(const ImageHolder<TYPE> &copy) = default;

    ImageHolder<TYPE> &operator=(ImageHolder<TYPE> &&assign) noexcept = default;

    void flip_v() {
      for (int y = 0; y < metadata.height / 2; y++) {
        for (int x = 0; x < metadata.width; x++) {
          int cur = (y * metadata.width + x) * metadata.channels;
          int inv = ((metadata.height - 1 - y) * metadata.width + x) * metadata.channels;
          for (int k = 0; k < metadata.channels; k++)
            std::swap(data()[cur + k], data()[inv + k]);
        }
      }
    }

    void flip_u() {
      for (int y = 0; y < metadata.height; y++) {
        for (int x = 0; x < metadata.width / 2; x++) {
          int cur = (y * metadata.width + x) * metadata.channels;
          int inv = (y * metadata.width + (metadata.width - 1 - x)) * metadata.channels;
          for (int k = 0; k < metadata.channels; k++)
            std::swap(data()[cur + k], data()[inv + k]);
        }
      }
    }

    void clear() { internal_data->clear(); }

    ImageHolderDataStorageInterface<TYPE> &data() { return *internal_data; }

    const ImageHolderDataStorageInterface<TYPE> &data() const { return *internal_data; }
  };

  /*******************************************************************************************************************************************/

  template<class TYPE>
  class ThumbnailImageHolder : public ImageHolder<TYPE> {
    using BASETYPE = ImageHolder<TYPE>;

   public:
    Thumbnail icon;
    ThumbnailImageHolder() = default;

    ThumbnailImageHolder(const axstd::span<TYPE> &assign,
                         const image::Metadata &_metadata_,
                         controller::ProgressStatus *status,
                         bool generate_thumbnail = true)
        : ImageHolder<TYPE>(assign, _metadata_) {
      if (generate_thumbnail)
        icon = Thumbnail(std::vector<TYPE>(BASETYPE::data().begin(), BASETYPE::data().end()), _metadata_, status);
    }

    ThumbnailImageHolder(const axstd::managed_vector<TYPE> &assign,
                         const image::Metadata &_metadata_,
                         controller::ProgressStatus *status,
                         bool generate_thumbnail = true)
        : ImageHolder<TYPE>(assign, _metadata_) {
      if (generate_thumbnail)
        icon = Thumbnail(std::vector<TYPE>(BASETYPE::data().begin(), BASETYPE::data().end()), _metadata_, status);
    }

    ThumbnailImageHolder(const std::vector<TYPE> &assign,
                         const image::Metadata &_metadata_,
                         controller::ProgressStatus *status,
                         bool generate_thumbnail = true)
        : ImageHolder<TYPE>(assign, _metadata_) {
      if (generate_thumbnail)
        icon = Thumbnail(std::vector<TYPE>(BASETYPE::data().begin(), BASETYPE::data().end()), _metadata_, status);
    }

    const QPixmap &thumbnail() { return icon.getIcon(); }

    std::string name() { return BASETYPE::metadata.name; }
  };

  /*******************************************************************************************************************************************/

}  // namespace image

#endif
