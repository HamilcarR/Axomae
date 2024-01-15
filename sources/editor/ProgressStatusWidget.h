#ifndef PROGRESSSTATUS_H
#define PROGRESSSTATUS_H
#include <QProgressBar>

class ProgressStatusWidget : public QProgressBar {
  Q_OBJECT
 public:
  ProgressStatusWidget(QWidget *parent = nullptr);
};

#endif
