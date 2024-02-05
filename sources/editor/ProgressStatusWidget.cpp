#include "ProgressStatusWidget.h"

ProgressStatusWidget::ProgressStatusWidget(QWidget *parent) : QProgressBar(parent) {}

void ProgressStatusWidget::display(std::string &format, int percentage) {
  setValue(percentage);
  setFormat(QString(std::string(format + std::to_string(percentage)).c_str()));
}