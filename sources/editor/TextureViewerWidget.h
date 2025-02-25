#ifndef TEXTUREVIEWERWIDGET_H
#define TEXTUREVIEWERWIDGET_H
#include "ContextMenuWidget.h"
#include "Image.h"
#include "metadata/RgbDisplayerLabel.h"
#include "ui_texture_viewer.h"
#include <internal/common/image/Rgb.h>

#include <QImage>
#include <QMainWindow>
#include <QResizeEvent>

namespace threading {
  class ThreadPool;
}

namespace controller {
  namespace event {
    class Event;
  }
  class Controller;
}  // namespace controller

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
  Q_OBJECT
 private:
  std::vector<float> raw_hdr_data{};

 public:
  explicit HdrTextureViewerWidget(const image::ImageHolder<float> &tex, QWidget *parent = nullptr);
  ~HdrTextureViewerWidget() override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void display(const std::vector<float> &image, int width, int height, int channels, bool color_corrected);
};
/*********************************************************************************************************************************************/

class HdrRenderViewerWidget : public QWidget {
  using EventManager = controller::event::Event;
  Q_OBJECT
 private:
  const image::ImageHolder<float> *target_buffer;  // Read only, need to be stored in a database , or in the main controller class.
  Ui::texture_viewer window{};
  std::unique_ptr<editor::RgbDisplayerLabel> label{};
  QImage image;
  std::string rgb_under_mouse_str{};
  std::unique_ptr<EventManager> widget_event_struct;
  std::unique_ptr<QTimer> timer;
  std::unique_ptr<ContextMenuWidget> context_menu;

 public:
  HdrRenderViewerWidget(const image::ImageHolder<float> *tex, controller::Controller *app_controller, QWidget *parent = nullptr);
  ~HdrRenderViewerWidget() override;
  HdrRenderViewerWidget(const HdrRenderViewerWidget &other) = delete;
  HdrRenderViewerWidget(HdrRenderViewerWidget &&other) noexcept = delete;
  HdrRenderViewerWidget &operator=(const HdrRenderViewerWidget &other) = delete;
  HdrRenderViewerWidget &operator=(HdrRenderViewerWidget &&other) noexcept = delete;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void closeEvent(QCloseEvent *event) override;

 signals:
  void viewerClosed(QWidget *widget_address);
  void onSaveRenderQuery(const image::ImageHolder<float> &target_buffer);
  void onStopRenderQuery();
 public slots:
  void updateImage();
};

#endif  // TEXTUREVIEWERWIDGET_H
