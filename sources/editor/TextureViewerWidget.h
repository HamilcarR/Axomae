#ifndef TEXTUREVIEWERWIDGET_H
#define TEXTUREVIEWERWIDGET_H
#include "Image.h"
#include "ui_texture_viewer.h"
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QResizeEvent>

class CustomLabelImageTex : public QLabel {
 public:
  explicit CustomLabelImageTex(std::string &current_rgb_text, QWidget *parent = nullptr);
  void paintEvent(QPaintEvent *event) override;

 private:
  std::string *current_rgb;
};

namespace controller::event {
  class Event;
}

class TextureViewerWidget : public QWidget {
 protected:
  using EventManager = controller::event::Event;

 public:
  explicit TextureViewerWidget(QWidget *parent = nullptr);
  ~TextureViewerWidget() override;
  void resizeEvent(QResizeEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void display(const std::vector<uint8_t> &image, int width, int height, int channels);

 protected:
  Ui::texture_viewer window{};
  std::unique_ptr<CustomLabelImageTex> label{};
  std::unique_ptr<QImage> image{};
  std::vector<uint8_t> raw_display_data{};
  std::string rgb_under_mouse_str{};
  image::Metadata metadata{};
  std::unique_ptr<EventManager> widget_event_struct;
};

class HdrTextureViewerWidget : public TextureViewerWidget {
 public:
  explicit HdrTextureViewerWidget(const image::ImageHolder<float> &tex, QWidget *parent = nullptr);
  void mouseMoveEvent(QMouseEvent *event) override;
  void display(const std::vector<float> &image, int width, int height, int channels, bool color_corrected);

 private:
  std::vector<float> raw_hdr_data{};
};

#endif  // TEXTUREVIEWERWIDGET_H
