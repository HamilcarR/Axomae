#ifndef NOVAVIEWERWIDGET_H
#define NOVAVIEWERWIDGET_H

#include "ui_nova_viewer.h"

#include <internal/macro/project_macros.h>

class QGridLayout;
class GLViewer;
namespace editor {
  class RgbDisplayerLabel;
}
class NovaViewerWidget : public QWidget {
  Q_OBJECT
 private:
  Ui::nova_viewer window{};
  std::unique_ptr<GLViewer> viewer;
  std::unique_ptr<QGridLayout> layout;
  std::unique_ptr<editor::RgbDisplayerLabel> rgb_label;

 public:
  explicit NovaViewerWidget(QWidget *parent = nullptr);
  ~NovaViewerWidget() override;
  NovaViewerWidget(const NovaViewerWidget &copy) = delete;
  NovaViewerWidget &operator=(const NovaViewerWidget &copy) = delete;
  NovaViewerWidget &operator=(NovaViewerWidget &&move) noexcept = delete;
  NovaViewerWidget(NovaViewerWidget &&move) noexcept = delete;
  ax_no_discard GLViewer *getViewer() const;
  void mouseMoveEvent(QMouseEvent *event) override;
};

#endif  // NovaViewerWidget_H
