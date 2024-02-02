#ifndef THUMBNAIL_H
#define THUMBNAIL_H
#include "Axomae_macros.h"
#include "IAxObject.h"
#include "OP_ProgressStatus.h"
#include "constants.h"
#include "image_utils.h"
#include <QImageWriter>
#include <QPixmap>
/**
 * @file ThumbnailList.h
 *
 */
template<typename T>
class Thumbnail : public IAxObject {
 private:
 public:
  Thumbnail() = default;
  ~Thumbnail() = default;
  // TODO : Implement functional test
  Thumbnail(std::vector<T> &rgb, int width, int height, int channels, bool needs_color_correct, controller::ProgressStatus *status = nullptr) {
    initProgress(status, "Generating Environment map thumbnail", static_cast<float>(width * height * channels));
    controller::ProgressManagerHelper helper(this);
    image_data = hdr_utils::hdr2image(rgb, width, height, channels, needs_color_correct, &helper);
    resetProgress();
    image = std::make_unique<QImage>(image_data.data(), width, height, QImage::Format_RGB32);
    *image = image->scaled(100, 50, Qt::KeepAspectRatio);
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
    image_data = std::move(move.image_data);
    return *this;
  }
  Thumbnail(Thumbnail &&move) noexcept {
    icon = std::move(move.icon);
    image = std::move(move.image);
    image_data = std::move(move.image_data);
  }

 private:
  std::unique_ptr<QPixmap> icon;
  std::unique_ptr<QImage> image;
  std::vector<uint8_t> image_data;
};

#endif