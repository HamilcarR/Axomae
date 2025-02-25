#ifndef RGBDISPLAYERLABEL_H
#define RGBDISPLAYERLABEL_H
#include <QLabel>
#include <internal/common/image/Rgb.h>
#include <internal/common/math/vector/Vector.h>

namespace editor {
  class RgbDisplayerLabel : public QLabel {
   public:
    struct DisplayInfo {
      image::Rgb texture_color_value;
      std::string rgb_display_text;
      Vec2ui pixel_position;
    };

   private:
    bool is_int;
    DisplayInfo current_pixel_display_info;

   public:
    explicit RgbDisplayerLabel(QWidget *parent = nullptr);
    void updateLabel(const float rgb[4], uint32_t pixel_x, uint32_t pixel_y, bool normalize = false);
    void updateLabel(const uint8_t rgb[4], uint32_t pixel_x, uint32_t pixel_y);

   protected:
    void paintEvent(QPaintEvent *event) override;
  };
}  // namespace editor
#endif  // RGBDISPLAYERLABEL_H
