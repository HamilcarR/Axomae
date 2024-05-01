#include "TextureViewerWidget.h"
#include "EventController.h"
#include "Logger.h"
#include "image_utils.h"

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