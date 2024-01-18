#ifndef THUMBNAIL_H
#define THUMBNAIL_H
#include "Axomae_macros.h"
#include "OP_ProgressStatus.h"
#include "constants.h"
#include <QImageWriter>
#include <QPixmap>
/**
 * @file ThumbnailList.h
 *
 */
template<typename T>
class Thumbnail {
 private:
  static constexpr float _gamma = 2.2;
  static constexpr float _exposure = 1.f;
  static constexpr float ratio = 1.f / 8.f;

  T tone_mapping_channel(T channel) { return 1.f - std::exp(-channel * _exposure); }
  T color_correction(T channel) { return std::pow(tone_mapping_channel(channel), 1.f / _gamma); }

 public:
  Thumbnail() = default;
  ~Thumbnail() = default;
  // TODO : Implement functional test
  Thumbnail(std::vector<T> &rgb, int width, int height, int channels, bool needs_color_correct, controller::ProgressStatus *status = nullptr) {
    ASSERT_IS_ARITHMETIC(T);
    assert(rgb.size() == static_cast<unsigned>(width * height * channels));
    int final_channels = channels == 3 ? 4 : channels;
    image_data.resize(width * height * final_channels);
    T max = 0;
    for (T i : rgb)
      max = std::max(max, i);
    assert(max > 0);
    if (needs_color_correct)
      max = color_correction(max);
    auto pbar_format = controller::progress_bar::generateData("Generating Environment map thumbnail", 0);
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int index = (i * width + j) * channels;
        if (status) {
          pbar_format.data.percentage = controller::progress_bar::computePercent((float)index, (float)(width * height * channels));
          status->op(&pbar_format);
        }
        T r, g, b;
        if (needs_color_correct) {
          r = (color_correction(rgb[index])) / max;
          g = (color_correction(rgb[index + 1])) / max;
          b = (color_correction(rgb[index + 2])) / max;
        } else {
          r = (rgb[index]) / max;
          g = (rgb[index + 1]) / max;
          b = (rgb[index + 2]) / max;
        }
        image_data[(i * width + j) * final_channels] = std::clamp(static_cast<int>(b * 255), 0, 255);
        image_data[(i * width + j) * final_channels + 1] = std::clamp(static_cast<int>(g * 255), 0, 255);
        image_data[(i * width + j) * final_channels + 2] = std::clamp(static_cast<int>(r * 255), 0, 255);
        image_data[(i * width + j) * final_channels + 3] = 0;
      }
    }
    image = std::make_unique<QImage>(image_data.data(), width, height, QImage::Format_RGB32);
    *image = image->scaled(100, 50, Qt::KeepAspectRatio);
    status->reset();
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