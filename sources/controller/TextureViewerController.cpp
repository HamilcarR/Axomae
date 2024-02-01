#include "TextureViewerController.h"
#include "image_utils.h"
#include <QPainter>

namespace controller {

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
  TextureViewerController::TextureViewerController(image::ImageHolder<float> &tex, QWidget *parent) : QWidget(parent), hdr_image(tex) {
    raw_data_uint = hdr_utils::hdr2image(tex.data, tex.metadata.width, tex.metadata.height, tex.metadata.channels, !tex.metadata.color_corrected);
    window.setupUi(this);
    image = std::make_unique<QImage>(raw_data_uint.data(), tex.metadata.width, tex.metadata.height, QImage::Format_RGB32);
    label = std::make_unique<CustomLabelImageTex>(current_rgb, this);
    label->setPixmap(QPixmap::fromImage(*image));
    label->setScaledContents(true);
    setMouseTracking(true);
  }
  void TextureViewerController::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    label->resize(event->size().width(), event->size().height());
  }

  template<class T>
  static QPoint mapToImage(const image::ImageHolder<T> &image, const QPoint &pos, int width, int height) {
    // Calculate the mapped position based on the image size and label size
    int imageX = pos.x() * image.metadata.width / width;
    int imageY = pos.y() * image.metadata.height / height;

    return QPoint(imageX, imageY);
  }
  void TextureViewerController::mouseMoveEvent(QMouseEvent *event) {
    QWidget::mouseMoveEvent(event);
    QPoint p = event->pos();

    if (p.x() < width() && p.y() < height() && p.x() >= 0 && p.y() >= 0) {
      p = mapToImage(hdr_image, p, width(), height());
      // QRgb rgb = image->pixel(p.x(), p.y());
      std::vector<float> &arr = hdr_image.data;
      unsigned int index = (p.y() * hdr_image.metadata.width + p.x()) * hdr_image.metadata.channels;
      float r_ = arr[index];
      float g_ = arr[index + 1];
      float b_ = arr[index + 2];
      std::string r = std::string("R : ") + std::to_string(r_);
      std::string g = std::string(" G : ") + std::to_string(g_);
      std::string b = std::string(" B : ") + std::to_string(b_);
      current_rgb = r + g + b;
      label->update();
    }
  }
}  // namespace controller