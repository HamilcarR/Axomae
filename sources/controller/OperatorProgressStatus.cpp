#include "OperatorProgressStatus.h"
#include "ProgressStatusWidget.h"
#include "internal/debug/Logger.h"

namespace controller {

  OperatorProgressStatus::OperatorProgressStatus(ProgressStatusWidget *progress_b) { module = progress_b; }

  static std::string formatted_string(const std::string &what, int percentage) { return what + ":" + std::to_string(percentage) + "%"; }

  bool OperatorProgressStatus::op(ioperator::OpData<progress_bar::ProgressBarTextFormat> *data) const {
    std::string format = formatted_string(data->data.format, data->data.percentage);
    module->setFormat(QString(format.c_str()));
    module->setValue(data->data.percentage);
    return true;
  }

  void OperatorProgressStatus::reset() const {
    module->setFormat(QString());
    module->setValue(0);
  }

  void OperatorProgressStatus::displayText(const std::string &message) {
    module->setValue(10);
    module->setFormat(QString(message.c_str()));
  }

  /******************************************************************************************************************************************/

  void IProgressManager::initProgress(ProgressStatus *status, const std::string &message_, float target_) {
    current = 0;
    progress_manager = status;
    message = message_;
    target = target_;
  }

  void IProgressManager::reset() {
    if (!progress_manager)
      return;
    progress_manager->reset();
  }

  void IProgressManager::initProgress(const std::string &message_, float target_) {
    current = 0;
    message = message_;
    target = target_;
  }
  void IProgressManager::displayStatusText(const std::string &message) {
    if (!progress_manager)
      return;
    progress_manager->displayText(message);
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
    current = 0;
    notifyProgress();
  }

  /******************************************************************************************************************************************/

  void ProgressManagerHelper::notifyProgress(float prog) {
    if (!progress_manager)
      return;
    progress_manager->setCurrent(prog);
    progress_manager->notifyProgress();
  }

  void ProgressManagerHelper::notifyProgress(PROGRESS_FLAG prog) {
    if (!progress_manager)
      return;
    float target = progress_manager->getTarget();
    switch (prog) {
      case ZERO:
        notifyProgress(0);
        break;
      case ONE_FOURTH:
        notifyProgress(target / 4);
        break;
      case TWO_FOURTH:
        notifyProgress(target / 2);
        break;
      case THREE_FOURTH:
        notifyProgress(target * 3 / 4);
        break;
      case COMPLETE:
        notifyProgress(target);
        break;

      default:
        break;
    }
  }

  void ProgressManagerHelper::reset() { notifyProgress(ZERO); }

}  // namespace controller