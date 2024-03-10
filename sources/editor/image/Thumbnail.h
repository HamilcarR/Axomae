#ifndef THUMBNAIL_H
#define THUMBNAIL_H
#include "Axomae_macros.h"
#include "IAxObject.h"
#include "Metadata.h"
#include "OP_ProgressStatus.h"
#include "constants.h"
#include "image_utils.h"
#include <QPixmap>
/**
 * @file ThumbnailList.h
 *
 */
class Thumbnail : public IAxObject {
 public:
  Thumbnail() = default;

  ~Thumbnail() = default;

  Thumbnail &operator=(const Thumbnail &copy) = delete;

  Thumbnail(const Thumbnail &copy) = delete;

  Thumbnail &operator=(Thumbnail &&move) noexcept;

  Thumbnail(Thumbnail &&move) noexcept;

  Thumbnail(std::vector<uint8_t> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status = nullptr);

  Thumbnail(std::vector<float> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status = nullptr);

  [[nodiscard]] const QPixmap &getIcon();

 private:
  std::unique_ptr<QPixmap> icon;
  std::unique_ptr<QImage> image;
};

#endif