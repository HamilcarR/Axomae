#ifndef THUMBNAIL_H
#define THUMBNAIL_H
#include "Metadata.h"
#include "OperatorProgressStatus.h"
#include "constants.h"
#include <QPixmap>
/**
 * @file ThumbnailList.h
 *
 */
class Thumbnail : public controller::IProgressManager {
 private:
  std::unique_ptr<QPixmap> icon;
  std::unique_ptr<QImage> image;

 public:
  Thumbnail() = default;
  ~Thumbnail() = default;
  Thumbnail &operator=(const Thumbnail &copy) = delete;
  Thumbnail(const Thumbnail &copy) = delete;
  Thumbnail &operator=(Thumbnail &&move) noexcept;
  Thumbnail(Thumbnail &&move) noexcept;
  Thumbnail(const std::vector<uint8_t> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status = nullptr);
  Thumbnail(const std::vector<float> &rgb, const image::Metadata &metadata, controller::ProgressStatus *status = nullptr);
  ax_no_discard const QPixmap &getIcon();
};

#endif
