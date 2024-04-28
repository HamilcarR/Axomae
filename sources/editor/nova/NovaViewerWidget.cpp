#include "NovaViewerWidget.h"
#include "GLViewer.h"
#include "NovaRenderer.h"
#include <QGridLayout>

NovaViewerWidget::NovaViewerWidget(QWidget *parent) : QWidget(parent) {
  window.setupUi(this);
  std::unique_ptr<IRenderer> r = std::make_unique<NovaRenderer>(this->width(), this->height(), nullptr);
  viewer = std::make_unique<GLViewer>(r, this);
  layout = std::make_unique<QGridLayout>(this);
  layout->addWidget(viewer.get());
  layout->setSpacing(0);
  viewer->setMouseTracking(true);
  setMouseTracking(true);
}
NovaViewerWidget::~NovaViewerWidget() = default;

GLViewer *NovaViewerWidget::getViewer() const { return viewer.get(); }
