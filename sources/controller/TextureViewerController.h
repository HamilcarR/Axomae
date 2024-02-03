#ifndef TEXTUREVIEWERCONTROLLER_H
#define TEXTUREVIEWERCONTROLLER_H
#include "Image.h"
#include "ui_texture_viewer.h"
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QResizeEvent>

namespace controller {

  class CustomLabelImageTex : public QLabel {
   public:
    explicit CustomLabelImageTex(std::string &current_rgb_text, QWidget *parent = nullptr);
    void paintEvent(QPaintEvent *event) override;

   private:
    std::string *current_rgb;
  };

  class TextureViewerController : public QWidget {
   public:
    TextureViewerController();
    TextureViewerController(image::ImageHolder<float> &tex, QWidget *parent = nullptr);
    TextureViewerController(const std::vector<uint8_t> &image, int width, int height, int channels, QWidget *parent = nullptr);
    void resizeEvent(QResizeEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;

   private:
    Ui::texture_viewer window{};
    std::unique_ptr<CustomLabelImageTex> label{};
    std::unique_ptr<QImage> image{};
    std::vector<uint8_t> raw_data_uint{};
    image::ImageHolder<float> hdr_image{};
    std::string current_rgb{};
  };

}  // namespace controller
#endif  // TEXTUREVIEWERCONTROLLER_H
