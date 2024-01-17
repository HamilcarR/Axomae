#ifndef OP_PROGRESSSTATUS_H
#define OP_PROGRESSSTATUS_H
#include "Operator.h"

class ProgressStatusWidget;
namespace controller {
  namespace progress_bar {

    struct ProgressBarTextFormat {
      std::string format;
      int percentage;
    };

    inline ProgressBarTextFormat generateData(std::string message, int percentage) {
      ProgressBarTextFormat format{};
      format.percentage = percentage;
      format.format = message;
      return format;
    }
  }  // namespace progress_bar
  class OP_ProgressStatus final : public ioperator::IOperator<ProgressStatusWidget, progress_bar::ProgressBarTextFormat> {

   public:
    explicit OP_ProgressStatus(ProgressStatusWidget *progress_bar);
    bool op(ioperator::OpData<progress_bar::ProgressBarTextFormat> *data) const override;
    void reset() const override;
  };

  using ProgressStatus = OP_ProgressStatus;
}  // namespace controller
#endif
