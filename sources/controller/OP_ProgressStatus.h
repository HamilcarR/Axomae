#ifndef OP_PROGRESSSTATUS_H
#define OP_PROGRESSSTATUS_H
#include "Operator.h"
#include "ProgressStatusWidget.h"

namespace controller::progress_bar {

  class OP_ProgressStatus final : public IOperator<ProgressStatusWidget> {

   public:
    explicit OP_ProgressStatus(ProgressStatusWidget *progress_bar);
    bool op() const override;
  };

  using ProgressStatus = OP_ProgressStatus;
}  // namespace controller::progress_bar

#endif
