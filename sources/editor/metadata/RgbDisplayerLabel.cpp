#include "RgbDisplayerLabel.h"

#include "internal/common/image/image_utils.h"

#include <QPainter>
namespace editor {
  constexpr int COLOR_DISPLAYER_SIZE = 30;
  RgbDisplayerLabel::RgbDisplayerLabel(QWidget *parent) : QLabel(parent) { setMouseTracking(true); }

  void RgbDisplayerLabel::paintEvent(QPaintEvent *event) {
    QLabel::paintEvent(event);
    QPainter text_painter(this);
    text_painter.setRenderHint(QPainter::Antialiasing);
    QFont font = text_painter.font();
    font.setPixelSize(12);
    text_painter.setFont(font);
    text_painter.fillRect(0, height() - 12, width(), 12, QColor(100, 100, 100, 200));
    text_painter.setPen(QColor(10, 10, 10, 225));

    text_painter.drawText(0, this->height() - 1, current_pixel_display_info.rgb_display_text.c_str());
    QFontMetrics metrics(font);
    int width_text = metrics.size(Qt::TextSingleLine, QString(current_pixel_display_info.rgb_display_text.c_str())).width();
    QPainter color_painter(this);
    QColor pixel_color_mouse;
    pixel_color_mouse.setRed((int)current_pixel_display_info.texture_color_value.red);
    pixel_color_mouse.setGreen((int)current_pixel_display_info.texture_color_value.green);
    pixel_color_mouse.setBlue((int)current_pixel_display_info.texture_color_value.blue);
    pixel_color_mouse.setAlpha(255);
    color_painter.fillRect(width_text + 1, height() - 12, COLOR_DISPLAYER_SIZE, 12, pixel_color_mouse);
  }

  void RgbDisplayerLabel::updateLabel(const float rgb[4], bool normalize) {
    current_pixel_display_info.texture_color_value = {rgb[0], rgb[1], rgb[2], rgb[3]};

    current_pixel_display_info.rgb_display_text = current_pixel_display_info.texture_color_value.to_string();
    current_pixel_display_info.texture_color_value.red = std::clamp(
        hdr_utils::color_correction(current_pixel_display_info.texture_color_value.red) * 255, 0.f, 255.f);
    current_pixel_display_info.texture_color_value.green = std::clamp(
        hdr_utils::color_correction(current_pixel_display_info.texture_color_value.green) * 255, 0.f, 255.f);
    current_pixel_display_info.texture_color_value.blue = std::clamp(
        hdr_utils::color_correction(current_pixel_display_info.texture_color_value.blue) * 255, 0.f, 255.f);
    current_pixel_display_info.texture_color_value.alpha *= 255;
    update();
  }

  void RgbDisplayerLabel::updateLabel(const uint8_t rgb[4]) {
    current_pixel_display_info.texture_color_value = {(float)rgb[0], (float)rgb[1], (float)rgb[2], (float)rgb[3]};

    current_pixel_display_info.rgb_display_text = current_pixel_display_info.texture_color_value.to_stringi();

    update();
  }

}  // namespace editor