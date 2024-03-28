#ifndef NOVAVIEWERWIDGET_H
#define NOVAVIEWERWIDGET_H
#include "GLViewer.h"
#include "ui_nova_viewer.h"

class QGridLayout;
class NovaViewerWidget : public QWidget {
  Q_OBJECT
 private:
  Ui::nova_viewer window{};
  std::unique_ptr<GLViewer> viewer;
  std::unique_ptr<QGridLayout> layout;

 public:
  explicit NovaViewerWidget(QWidget *parent = nullptr);
  ~NovaViewerWidget() override;
  NovaViewerWidget(const NovaViewerWidget &copy) = delete;
  NovaViewerWidget &operator=(const NovaViewerWidget &copy) = delete;
  NovaViewerWidget &operator=(NovaViewerWidget &&move) noexcept = delete;
  NovaViewerWidget(NovaViewerWidget &&move) noexcept = delete;
};

#endif  // NovaViewerWidget_H
