#include "TextureViewerWidget.h"
#include "Config.h"
#include "GUIWindow.h"
#include "event/EventController.h"
#include "internal/common/image/image_utils.h"
#include <QTimer>

static QPoint mapToImage(int image_width, int image_height, const QPoint &pos, int width, int height) {
  // Calculate the mapped position based on the image size and label size
  int imageX = pos.x() * image_width / width;
  int imageY = pos.y() * image_height / height;
  return {imageX, imageY};
}

/************************************************************************************************************************************************************/

TextureViewerWidget::TextureViewerWidget(QWidget *parent) : QWidget(parent) {
  window.setupUi(this);
  label = std::make_unique<editor::RgbDisplayerLabel>(this);
  label->setScaledContents(true);
  widget_event_struct = std::make_unique<EventManager>();
  setMouseTracking(true);
}
TextureViewerWidget::~TextureViewerWidget() = default;

void TextureViewerWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  if (label)
    label->resize(event->size().width(), event->size().height());
}

void TextureViewerWidget::display(const std::vector<uint8_t> &img, int width, int height, int channels) {
  raw_display_data = img;
  metadata.width = width;
  metadata.height = height;
  metadata.channels = channels;
  if (channels == 3)
    image = std::make_unique<QImage>(raw_display_data.data(), width, height, QImage::Format_RGB888);
  else if (channels == 4)
    image = std::make_unique<QImage>(raw_display_data.data(), width, height, QImage::Format_RGBA8888);
  AX_ASSERT(channels == 3 || channels == 4, "Invalid channel number.");
  label->setPixmap(QPixmap::fromImage(*image));
  label->setScaledContents(true);
}

void TextureViewerWidget::mouseMoveEvent(QMouseEvent *event) {
  QWidget::mouseMoveEvent(event);
  QPoint p = event->pos();
  widget_event_struct->flag |= EventManager::EVENT_MOUSE_MOVE;
  if (p.x() < width() && p.y() < height() && p.x() >= 0 && p.y() >= 0) {
    p = mapToImage(static_cast<int>(metadata.width), static_cast<int>(metadata.height), p, width(), height());
    unsigned int index = (p.y() * metadata.width + p.x()) * metadata.channels;
    if (!raw_display_data.empty()) {
      uint8_t r_ = raw_display_data[index];
      uint8_t g_ = raw_display_data[index + 1];
      uint8_t b_ = raw_display_data[index + 2];
      uint8_t a_ = 255;
      const uint8_t rgb[4] = {r_, g_, b_, a_};
      label->updateLabel(rgb);
    }
  }
  widget_event_struct->flag &= ~EventManager::EVENT_MOUSE_MOVE;
}

/************************************************************************************************************************************************************/

HdrTextureViewerWidget::HdrTextureViewerWidget(const image::ImageHolder<float> &tex, QWidget *parent)
    : TextureViewerWidget(parent), raw_hdr_data(tex.data) {
  raw_display_data = hdr_utils::hdr2image(tex.data, tex.metadata.width, tex.metadata.height, tex.metadata.channels, !tex.metadata.color_corrected);
  window.setupUi(this);
  metadata.width = tex.metadata.width;
  metadata.height = tex.metadata.height;
  metadata.channels = tex.metadata.channels;
  metadata.color_corrected = tex.metadata.color_corrected;
  image = std::make_unique<QImage>(raw_display_data.data(), tex.metadata.width, tex.metadata.height, QImage::Format_RGB32);
  label = std::make_unique<editor::RgbDisplayerLabel>(this);
  label->setPixmap(QPixmap::fromImage(*image));
  label->setScaledContents(true);
  setMouseTracking(true);
}

HdrTextureViewerWidget::~HdrTextureViewerWidget() = default;

void HdrTextureViewerWidget::display(const std::vector<float> &img, int width, int height, int channels, bool color_corrected) {
  raw_display_data = hdr_utils::hdr2image(img, width, height, channels, !color_corrected);
  if (channels == 3)
    image = std::make_unique<QImage>(raw_display_data.data(), width, height, QImage::Format_RGB888);
  else if (channels == 4)
    image = std::make_unique<QImage>(raw_display_data.data(), width, height, QImage::Format_RGBA8888);
  AX_ASSERT(channels == 3 || channels == 4, "Invalid channel number.");
  if (!label)
    label = std::make_unique<editor::RgbDisplayerLabel>(this);
  label->setPixmap(QPixmap::fromImage(*image));
}

void HdrTextureViewerWidget::mouseMoveEvent(QMouseEvent *event) {
  QWidget::mouseMoveEvent(event);
  QPoint p = event->pos();

  widget_event_struct->flag |= EventManager::EVENT_MOUSE_MOVE;
  if (p.x() < width() && p.y() < height() && p.x() >= 0 && p.y() >= 0) {
    p = mapToImage(static_cast<int>(metadata.width), static_cast<int>(metadata.height), p, width(), height());
    unsigned int index = (p.y() * metadata.width + p.x()) * metadata.channels;
    if (!raw_hdr_data.empty()) {
      float r_ = raw_hdr_data[index];
      float g_ = raw_hdr_data[index + 1];
      float b_ = raw_hdr_data[index + 2];
      float a_ = 1.f;
      const float rgb[4] = {r_, g_, b_, a_};
      label->updateLabel(rgb, false);
    }
  }
  widget_event_struct->flag &= ~EventManager::EVENT_MOUSE_MOVE;
}

/************************************************************************************************************************************************************/

HdrRenderViewerWidget::HdrRenderViewerWidget(const image::ImageHolder<float> *tex, controller::Controller *app_controller, QWidget *parent)
    : QWidget(parent) {

  window.setupUi(this);
  widget_event_struct = std::make_unique<EventManager>();
  target_buffer = tex;
  timer = std::make_unique<QTimer>();
  QObject::connect(timer.get(), &QTimer::timeout, this, &HdrRenderViewerWidget::updateImage);
  QObject::connect(this, &HdrRenderViewerWidget::viewerClosed, app_controller, &controller::Controller::slot_on_closed_spawn_window);
  QObject::connect(this, &HdrRenderViewerWidget::onSaveRenderQuery, app_controller, &controller::Controller::slot_nova_save_bake);
  QObject::connect(this, &HdrRenderViewerWidget::onStopRenderQuery, app_controller, &controller::Controller::slot_nova_stop_bake);
  label = std::make_unique<editor::RgbDisplayerLabel>(this);
  label->setScaledContents(true);
  context_menu = std::make_unique<ContextMenuWidget>(this);

  setMouseTracking(true);

  timer->start(1000);
}

HdrRenderViewerWidget::~HdrRenderViewerWidget() = default;

void HdrRenderViewerWidget::mouseMoveEvent(QMouseEvent *event) {
  QWidget::mouseMoveEvent(event);
  if (!target_buffer)
    return;
  QPoint p = event->pos();

  widget_event_struct->flag |= EventManager::EVENT_MOUSE_MOVE;
  if (p.x() < width() && p.y() < height() && p.x() >= 0 && p.y() >= 0) {
    p = mapToImage(static_cast<int>(target_buffer->metadata.width), static_cast<int>(target_buffer->metadata.height), p, width(), height());
    unsigned int index = (p.y() * target_buffer->metadata.width + p.x()) * target_buffer->metadata.channels;
    if (!target_buffer->data.empty()) {
      float r_ = target_buffer->data[index];
      float g_ = target_buffer->data[index + 1];
      float b_ = target_buffer->data[index + 2];
      float a_ = 1.f;
      const float rgb[4] = {r_, g_, b_, a_};
      label->updateLabel(rgb, false);
    }
  }
  widget_event_struct->flag &= ~EventManager::EVENT_MOUSE_MOVE;
}
void HdrRenderViewerWidget::mouseReleaseEvent(QMouseEvent *event) {
  QWidget::mouseReleaseEvent(event);
  QPoint p = event->pos();
  if (event->button() == Qt::RightButton) {
    widget_event_struct->flag |= EventManager::EVENT_MOUSE_R_RELEASE;
    ContextMenuWidget::ACTION action = context_menu->spawnMenuOnPos(p);
    if (action == ContextMenuWidget::SAVE) {
      if (target_buffer)
        emit onSaveRenderQuery(*target_buffer);
    } else if (action == ContextMenuWidget::STOP) {
      emit onStopRenderQuery();
    }
    widget_event_struct->flag &= ~EventManager::EVENT_MOUSE_R_RELEASE;
  }
}

void HdrRenderViewerWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  if (label)
    label->resize(event->size().width(), event->size().height());
}
void HdrRenderViewerWidget::closeEvent(QCloseEvent *event) {
  QWidget::closeEvent(event);
  emit viewerClosed(this);
}

void HdrRenderViewerWidget::updateImage() {
  AX_ASSERT_EQ(target_buffer->data.size(), target_buffer->metadata.width * target_buffer->metadata.height * target_buffer->metadata.channels);

  std::vector<uint8_t> normalized = hdr_utils::hdr2image(target_buffer->data,
                                                         target_buffer->metadata.width,
                                                         target_buffer->metadata.height,
                                                         target_buffer->metadata.channels,
                                                         !target_buffer->metadata.color_corrected);

  image = QImage(normalized.data(), target_buffer->metadata.width, target_buffer->metadata.height, QImage::Format_ARGB32);
  label->setPixmap(QPixmap::fromImage(image));
}
