#ifndef RGBDISPLAYERLABEL_H
#define RGBDISPLAYERLABEL_H
#include "Rgb.h"

#include <QLabel>
namespace editor {
  class RgbDisplayerLabel : public QLabel {
   public:
    struct DisplayInfo {

      image::Rgb texture_color_value;

    };

   private:
    DisplayInfo current_pixel_display_info;

   public:
    explicit RgbDisplayerLabel(QWidget *parent = nullptr);
    void updateLabel(const DisplayInfo &display_info, bool normalize = false);

   protected:
    void paintEvent(QPaintEvent *event) override;
  };
}  // namespace editor
#endif  // RGBDISPLAYERLABEL_H
