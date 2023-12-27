#include "Thumbnail.h"

const float _gamma = 2.2;
const float _exposure = 1.f;
const float ratio = 1.f / 8.f;
float tone_mapping_channel(float channel) { return 1.f - std::exp(-channel * _exposure); }
float color_correction(float channel) { return powf(tone_mapping_channel(channel), 1.f / _gamma); }

// TODO : Implement functional test
Thumbnail::Thumbnail(std::vector<float> &rgb, int width, int height, int channels, bool /*normalize*/, int icon_width, int icon_height) {
  assert(rgb.size() == static_cast<unsigned>(width * height * channels));
  QImage image(width, height, QImage::Format_RGB32);
  float max = 0;
  for (float i : rgb)
    max = std::max(max, i);
  assert(max > 0);
  max = color_correction(max);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {

      float r = (color_correction(rgb[(i * width + j) * channels])) / max;
      float g = (color_correction(rgb[(i * width + j) * channels + 1])) / max;
      float b = (color_correction(rgb[(i * width + j) * channels + 2])) / max;

      QColor color(static_cast<int>(r * 255.f), static_cast<int>(g * 255.f), static_cast<int>(b * 255.f), 255);
      image.setPixelColor(j, i, color);
    }
  }

  image = image.scaled(100, 50, Qt::KeepAspectRatio);
  icon = QPixmap::fromImage(image);
}
