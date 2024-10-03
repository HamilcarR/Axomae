#ifndef MESSAGEBOXWIDGET_H
#define MESSAGEBOXWIDGET_H
#include "internal/macro/project_macros.h"
#include <QMessageBox>

class WarningBoxWidget : public QMessageBox {

 public:
  CLASS_OCM(WarningBoxWidget)
  explicit WarningBoxWidget(const std::string &message, QWidget *parent = nullptr);
};

class InfoBoxWidget : public QMessageBox {

 public:
  CLASS_OCM(InfoBoxWidget)
  explicit InfoBoxWidget(const std::string &message, QWidget *parent = nullptr);
};

class CriticalBoxWidget : public QMessageBox {

 public:
  CLASS_OCM(CriticalBoxWidget)
  explicit CriticalBoxWidget(const std::string &message, QWidget *parent = nullptr);
};
#endif  // MESSAGEBOXWIDGET_H
