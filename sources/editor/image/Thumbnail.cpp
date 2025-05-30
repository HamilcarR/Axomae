#include "Thumbnail.h"
#include "internal/common/image/image_utils.h"
#include <QImageWriter>
Thumbnail::Thumbnail(const std::vector<uint8_t> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status) {
  initProgress(status, "Generating image thumbnail", static_cast<float>(metadata.width * metadata.height * metadata.channels));
  controller::ProgressManagerHelper helper(this);
  image = std::make_unique<QImage>(rgb.data(), metadata.width, metadata.height, QImage::Format_ARGB32);
  *image = image->scaled(100, 50, Qt::KeepAspectRatio);
  resetProgress();
}

Thumbnail::Thumbnail(const std::vector<float> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status) {
  initProgress(status, "Generating image thumbnail", static_cast<float>(metadata.width * metadata.height * metadata.channels));
  controller::ProgressManagerHelper helper(this);
  helper.notifyProgress(controller::ProgressManagerHelper::ONE_FOURTH);
  /*If already color corrected , we just pass false*/
  std::vector<uint8_t> tmp = hdr_utils::hdr2image(
      axstd::span<const float>(rgb), metadata.width, metadata.height, metadata.channels, !metadata.color_corrected);
  image = std::make_unique<QImage>(tmp.data(), metadata.width, metadata.height, QImage::Format_RGB32);
  *image = image->scaled(100, 50, Qt::KeepAspectRatio);
  resetProgress();
}

const QPixmap &Thumbnail::getIcon() {
  if (!icon)
    icon = std::make_unique<QPixmap>();
  *icon = QPixmap::fromImage(*image);
  return *icon;
}

Thumbnail &Thumbnail::operator=(Thumbnail &&move) noexcept {
  icon = std::move(move.icon);
  image = std::move(move.image);
  return *this;
}

Thumbnail::Thumbnail(Thumbnail &&move) noexcept {
  icon = std::move(move.icon);
  image = std::move(move.image);
}
