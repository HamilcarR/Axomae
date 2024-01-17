#ifndef PROGRESSSTATUS_H
#define PROGRESSSTATUS_H
#include "constants.h"
#include <QProgressBar>
class ProgressStatusWidget : public QProgressBar {
  Q_OBJECT
 public:
  ProgressStatusWidget(QWidget *parent = nullptr);

  /**
   * @brief Displays the progress bar as : {format_text} : {percentage}%
   *
   */
  void display(std::string &format_text, int percentage);
};

#endif
