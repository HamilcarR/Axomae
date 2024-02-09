#ifndef THUMBNAIL_H
#define THUMBNAIL_H
#include "Axomae_macros.h"
#include "IAxObject.h"
#include "Metadata.h"
#include "OP_ProgressStatus.h"
#include "constants.h"
#include "image_utils.h"
#include <QImageWriter>
#include <QPixmap>
/**
 * @file ThumbnailList.h
 *
 */
class Thumbnail : public IAxObject {
 private:
 public:
  Thumbnail() = default;
  ~Thumbnail() = default;

  Thumbnail(std::vector<uint8_t> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status = nullptr) {
    initProgress(status, "Generating image thumbnail", static_cast<float>(metadata.width * metadata.height * metadata.channels));
    controller::ProgressManagerHelper helper(this);
    image = std::make_unique<QImage>(rgb.data(), metadata.width, metadata.height, QImage::Format_ARGB32);
    *image = image->scaled(100, 50, Qt::KeepAspectRatio);
    resetProgress();
  }

  Thumbnail(std::vector<float> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status = nullptr) {
    initProgress(status, "Generating image thumbnail", static_cast<float>(metadata.width * metadata.height * metadata.channels));
    controller::ProgressManagerHelper helper(this);
    /*If already color corrected , we just pass false*/
    std::vector<uint8_t> tmp = hdr_utils::hdr2image(rgb, metadata.width, metadata.height, metadata.channels, !metadata.color_corrected, &helper);
    image = std::make_unique<QImage>(tmp.data(), metadata.width, metadata.height, QImage::Format_RGB32);
    *image = image->scaled(100, 50, Qt::KeepAspectRatio);
    resetProgress();
  }

  [[nodiscard]] const QPixmap &getIcon() {
    if (!icon)
      icon = std::make_unique<QPixmap>();
    *icon = QPixmap::fromImage(*image);
    return *icon;
  }

  Thumbnail &operator=(const Thumbnail &copy) = delete;
  Thumbnail(const Thumbnail &copy) = delete;
  Thumbnail &operator=(Thumbnail &&move) noexcept {
    icon = std::move(move.icon);
    image = std::move(move.image);
    return *this;
  }
  Thumbnail(Thumbnail &&move) noexcept {
    icon = std::move(move.icon);
    image = std::move(move.image);
  }

 private:
  std::unique_ptr<QPixmap> icon;
  std::unique_ptr<QImage> image;
};

#endif