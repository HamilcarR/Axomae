#include "TextureViewerWidget.h"
#include "image_utils.h"
#include <QPainter>

static QPoint mapToImage(int image_width, int image_height, const QPoint &pos, int width, int height) {
  // Calculate the mapped position based on the image size and label size
  int imageX = pos.x() * image_width / width;
  int imageY = pos.y() * image_height / height;
  return QPoint(imageX, imageY);
}

CustomLabelImageTex::CustomLabelImageTex(std::string &current_, QWidget *parent) : QLabel(parent) {
  current_rgb = &current_;
  setMouseTracking(true);
}

void CustomLabelImageTex::paintEvent(QPaintEvent *event) {
  QLabel::paintEvent(event);
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  QFont font = painter.font();
  font.setPixelSize(12);
  painter.setFont(font);
  painter.fillRect(0, height() - 12, width(), 12, QColor(100, 100, 100, 200));
  painter.setPen(QColor(10, 10, 10, 225));
  painter.drawText(0, this->height() - 1, current_rgb->c_str());
}

/************************************************************************************************************************************************************/

TextureViewerWidget::TextureViewerWidget(QWidget *parent) : QWidget(parent) {
  window.setupUi(this);
  label = std::make_unique<CustomLabelImageTex>(rgb_under_mouse_str, this);
  label->setScaledContents(true);
  setMouseTracking(true);
}

TextureViewerWidget::TextureViewerWidget(const std::vector<uint8_t> &image_, int width, int height, int channels, QWidget *parent)
    : QWidget(parent), raw_display_data(image_) {
  window.setupUi(this);
  metadata.width = width;
  metadata.height = height;
  metadata.channels = channels;
  display(image_, width, height, channels);
  label->setScaledContents(true);
  setMouseTracking(true);
}

TextureViewerWidget::TextureViewerWidget(const image::ImageHolder<uint8_t> &tex, QWidget *parent)
    : QWidget(parent), raw_display_data(tex.data), metadata(tex.metadata) {
  window.setupUi(this);

  image = std::make_unique<QImage>(raw_display_data.data(), tex.metadata.width, tex.metadata.height, QImage::Format_RGB32);
  label = std::make_unique<CustomLabelImageTex>(rgb_under_mouse_str, this);
  label->setPixmap(QPixmap::fromImage(*image));
  label->setScaledContents(true);
  setMouseTracking(true);
}

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
  assert(channels == 3 || channels == 4);  // Other cases for later
  label->setPixmap(QPixmap::fromImage(*image));
  label->setScaledContents(true);
}

void TextureViewerWidget::mouseMoveEvent(QMouseEvent *event) {
  QWidget::mouseMoveEvent(event);
  QPoint p = event->pos();

  if (p.x() < width() && p.y() < height() && p.x() >= 0 && p.y() >= 0) {
    p = mapToImage(static_cast<int>(metadata.width), static_cast<int>(metadata.height), p, width(), height());
    unsigned int index = (p.y() * metadata.width + p.x()) * metadata.channels;
    if (!raw_display_data.empty()) {
      uint8_t r_ = raw_display_data[index];
      uint8_t g_ = raw_display_data[index + 1];
      uint8_t b_ = raw_display_data[index + 2];
      std::string r = std::string("R : ") + std::to_string(r_);
      std::string g = std::string(" G : ") + std::to_string(g_);
      std::string b = std::string(" B : ") + std::to_string(b_);
      rgb_under_mouse_str = r + g + b;
    }
    label->update();
  }
}

/************************************************************************************************************************************************************/

HdrTextureViewerWidget::HdrTextureViewerWidget(QWidget *parent) : TextureViewerWidget(parent) {
  window.setupUi(this);
  label = std::make_unique<CustomLabelImageTex>(rgb_under_mouse_str, this);
}

HdrTextureViewerWidget::HdrTextureViewerWidget(
    const std::vector<float> &image_, int width, int height, int channels, bool color_corrected, QWidget *parent)
    : TextureViewerWidget(parent), raw_hdr_data(image_) {
  window.setupUi(this);
  display(image_, width, height, channels, color_corrected);
  label->setScaledContents(true);
  setMouseTracking(true);
}

HdrTextureViewerWidget::HdrTextureViewerWidget(const image::ImageHolder<float> &tex, QWidget *parent)
    : TextureViewerWidget(parent), raw_hdr_data(tex.data) {
  raw_display_data = hdr_utils::hdr2image(tex.data, tex.metadata.width, tex.metadata.height, tex.metadata.channels, !tex.metadata.color_corrected);
  window.setupUi(this);
  metadata.width = tex.metadata.width;
  metadata.height = tex.metadata.height;
  metadata.channels = tex.metadata.channels;
  metadata.color_corrected = tex.metadata.color_corrected;
  image = std::make_unique<QImage>(raw_display_data.data(), tex.metadata.width, tex.metadata.height, QImage::Format_RGB32);
  label = std::make_unique<CustomLabelImageTex>(rgb_under_mouse_str, this);
  label->setPixmap(QPixmap::fromImage(*image));
  label->setScaledContents(true);
  setMouseTracking(true);
}

void HdrTextureViewerWidget::display(const std::vector<float> &img, int width, int height, int channels, bool color_corrected) {
  raw_display_data = hdr_utils::hdr2image(img, width, height, channels, !color_corrected);
  if (channels == 3)
    image = std::make_unique<QImage>(raw_display_data.data(), width, height, QImage::Format_RGB888);
  else if (channels == 4)
    image = std::make_unique<QImage>(raw_display_data.data(), width, height, QImage::Format_RGBA8888);
  assert(channels == 3 || channels == 4);
  if (!label)
    label = std::make_unique<CustomLabelImageTex>(rgb_under_mouse_str, this);
  label->setPixmap(QPixmap::fromImage(*image));
}

void HdrTextureViewerWidget::mouseMoveEvent(QMouseEvent *event) {
  QWidget::mouseMoveEvent(event);
  QPoint p = event->pos();

  if (p.x() < width() && p.y() < height() && p.x() >= 0 && p.y() >= 0) {
    p = mapToImage(static_cast<int>(metadata.width), static_cast<int>(metadata.height), p, width(), height());
    // QRgb rgb = image->pixel(p.x(), p.y());
    unsigned int index = (p.y() * metadata.width + p.x()) * metadata.channels;
    if (!raw_hdr_data.empty()) {
      float r_ = raw_hdr_data[index];
      float g_ = raw_hdr_data[index + 1];
      float b_ = raw_hdr_data[index + 2];
      std::string r = std::string("R : ") + std::to_string(r_);
      std::string g = std::string(" G : ") + std::to_string(g_);
      std::string b = std::string(" B : ") + std::to_string(b_);
      rgb_under_mouse_str = r + g + b;
    }
    label->update();
  }
}