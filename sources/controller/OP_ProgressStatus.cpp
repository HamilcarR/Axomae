#include "OP_ProgressStatus.h"
#include "Logger.h"
#include "ProgressStatusWidget.h"

namespace controller {

  OP_ProgressStatus::OP_ProgressStatus(ProgressStatusWidget *progress_b) { module = progress_b; }

  static std::string formatted_string(std::string &what, int percentage) { return what + ":" + std::to_string(percentage) + "%"; }

  bool OP_ProgressStatus::op(ioperator::OpData<progress_bar::ProgressBarTextFormat> *data) const {
    std::string format = formatted_string(data->data.format, data->data.percentage);
    module->setFormat(QString(format.c_str()));
    module->setValue(data->data.percentage);
    return true;
  }

  void OP_ProgressStatus::reset() const {
    module->setFormat(QString());
    module->setValue(0);
  }
}  // namespace controller