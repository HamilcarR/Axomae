#include "MessageBoxWidget.h"

WarningBoxWidget::WarningBoxWidget(const std::string &message, QWidget *parent) : QMessageBox(parent) {
  setText(message.c_str());
  setIcon(QMessageBox::Warning);
  setWindowTitle("Warning");
  setStandardButtons(QMessageBox::Close);
}

InfoBoxWidget::InfoBoxWidget(const std::string &message, QWidget *parent) : QMessageBox(parent) {
  setText(message.c_str());
  setIcon(QMessageBox::Information);
  setWindowTitle("Info");
  setStandardButtons(QMessageBox::Close);
}

CriticalBoxWidget::CriticalBoxWidget(const std::string &message, QWidget *parent) : QMessageBox(parent) {
  setText(message.c_str());
  setIcon(QMessageBox::Critical);
  setWindowTitle("Critical");
  setStandardButtons(QMessageBox::Close);
}
