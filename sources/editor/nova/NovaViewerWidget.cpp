#include "NovaViewerWidget.h"
#include "GLViewer.h"
#include "NovaRenderer.h"
#include "metadata/RgbDisplayerLabel.h"

#include <QGridLayout>

NovaViewerWidget::NovaViewerWidget(QWidget *parent) : QWidget(parent) {
  window.setupUi(this);
  std::unique_ptr<IRenderer> r = std::make_unique<NovaRenderer>(this->width(), this->height(), nullptr);
  layout = std::make_unique<QGridLayout>(this);
  layout->setSpacing(0);

  viewer = std::make_unique<GLViewer>(r, this);
  viewer->setMouseTracking(true);
  viewer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  layout->addWidget(viewer.get(), 0, 0);

  rgb_label = std::make_unique<editor::RgbDisplayerLabel>();
  rgb_label->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  layout->addWidget(rgb_label.get(), 1, 0);

  setMouseTracking(true);
}
NovaViewerWidget::~NovaViewerWidget() = default;

GLViewer *NovaViewerWidget::getViewer() const { return viewer.get(); }

void NovaViewerWidget::mouseMoveEvent(QMouseEvent *event) {
  QWidget::mouseMoveEvent(event);
  if (!viewer)
    return;
  if (!rgb_label)
    return;
  QPoint p = this->mapFromGlobal(QCursor::pos());
  image::Rgb framebuffer_color_pixel = viewer->getFramebufferColor(p.x(), p.y());
  rgb_label->updateLabel({framebuffer_color_pixel}, true);
}
