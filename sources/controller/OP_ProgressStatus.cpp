#include "OP_ProgressStatus.h"
#include "Logger.h"
#include "ProgressStatusWidget.h"

namespace controller {

  OP_ProgressStatus::OP_ProgressStatus(ProgressStatusWidget *progress_b) { module = progress_b; }

  static std::string formatted_string(const std::string &what, int percentage) { return what + ":" + std::to_string(percentage) + "%"; }

  bool OP_ProgressStatus::op(ioperator::OpData<progress_bar::ProgressBarTextFormat> *data) const {
    std::string format = formatted_string(data->data.format, data->data.percentage);
    Mutex::Lock lock(mutex);
    module->setFormat(QString(format.c_str()));
    module->setValue(data->data.percentage);
    return true;
  }

  void OP_ProgressStatus::reset() const {
    module->setFormat(QString());
    module->setValue(0);
  }

  /******************************************************************************************************************************************/

  void IProgressManager::initProgress(ProgressStatus *status, const std::string &message_, float target_) {
    current = 0;
    progress_manager = status;
    message = message_;
    target = target_;
  }

  void IProgressManager::initProgress(const std::string &message_, float target_) {
    current = 0;
    message = message_;
    target = target_;
  }

  void IProgressManager::notifyProgress() {
    if (!progress_manager)
      return;
    progress_bar::ProgressBarTextFormat prog_data;
    prog_data.format = message;
    prog_data.percentage = progress_bar::computePercent(current, target);
    ioperator::OpData<progress_bar::ProgressBarTextFormat> opdata(prog_data);
    progress_manager->op(&opdata);
  }

  void IProgressManager::resetProgress() {
    message = "";
    current = -1;
  }

  /******************************************************************************************************************************************/

  void ProgressManagerHelper::notifyProgress(float prog) {
    if (!progress_helpee)
      return;
    float &rf = progress_helpee->getCurrentRefVar();
    rf = prog;
    progress_helpee->notifyProgress();
  }

}  // namespace controller