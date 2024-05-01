#ifndef TEXTUREVIEWERWIDGET_H
#define TEXTUREVIEWERWIDGET_H
#include "Image.h"
#include "Rgb.h"
#include "metadata/RgbDisplayerLabel.h"
#include "ui_texture_viewer.h"
#include <QImage>
#include <QMainWindow>
#include <QResizeEvent>

namespace controller::event {
  class Event;
}

class TextureViewerWidget : public QWidget {
 protected:
  using EventManager = controller::event::Event;

 protected:
  Ui::texture_viewer window{};
  std::unique_ptr<editor::RgbDisplayerLabel> label{};
  std::unique_ptr<QImage> image{};
  std::vector<uint8_t> raw_display_data{};
  std::string rgb_under_mouse_str{};
  image::Metadata metadata{};
  std::unique_ptr<EventManager> widget_event_struct;

 public:
  explicit TextureViewerWidget(QWidget *parent = nullptr);
  ~TextureViewerWidget() override;
  TextureViewerWidget(const TextureViewerWidget &copy) = delete;
  TextureViewerWidget(TextureViewerWidget &&move) noexcept = delete;
  TextureViewerWidget &operator=(const TextureViewerWidget &copy) = delete;
  TextureViewerWidget &operator=(TextureViewerWidget &&move) noexcept = delete;
  void resizeEvent(QResizeEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void display(const std::vector<uint8_t> &image, int width, int height, int channels);
};

class HdrTextureViewerWidget : public TextureViewerWidget {
 private:
  std::vector<float> raw_hdr_data{};

 public:
  explicit HdrTextureViewerWidget(const image::ImageHolder<float> &tex, QWidget *parent = nullptr);
  ~HdrTextureViewerWidget() override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void display(const std::vector<float> &image, int width, int height, int channels, bool color_corrected);
};

#endif  // TEXTUREVIEWERWIDGET_H
