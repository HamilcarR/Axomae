#ifndef THUMBNAIL_H
#define THUMBNAIL_H
#include "Axomae_macros.h"
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

  // TODO : Implement functional test
  Thumbnail(std::vector<T> &rgb, int width, int height, int channels, bool needs_color_correct) {
    ASSERT_IS_ARITHMETIC(T);
    assert(rgb.size() == static_cast<unsigned>(width * height * channels));
    // QImage image(width, height, QImage::Format_RGB32);
    image = std::make_unique<QImage>(width, height, QImage::Format_RGB32);
    T max = 0;
    for (T i : rgb)
      max = std::max(max, i);
    assert(max > 0);
    if (needs_color_correct)
      max = color_correction(max);
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        T r, g, b;
        if (needs_color_correct) {
          r = (color_correction(rgb[(i * width + j) * channels])) / max;
          g = (color_correction(rgb[(i * width + j) * channels + 1])) / max;
          b = (color_correction(rgb[(i * width + j) * channels + 2])) / max;
        } else {
          r = (rgb[(i * width + j) * channels]) / max;
          g = (rgb[(i * width + j) * channels + 1]) / max;
          b = (rgb[(i * width + j) * channels + 2]) / max;
        }
        QColor color(static_cast<int>(r * 255.f), static_cast<int>(g * 255.f), static_cast<int>(b * 255.f), 255);
        image->setPixelColor(j, i, color);
      }
    }
    *image = image->scaled(100, 50, Qt::KeepAspectRatio);
  }

  [[nodiscard]] const QPixmap &getIcon() {
    if (!icon)
      icon = std::make_unique<QPixmap>();
    *icon = QPixmap::fromImage(*image);
    return *icon;
  }

 private:
  std::unique_ptr<QPixmap> icon;
  std::unique_ptr<QImage> image;
};

#endif