#ifndef OP_PROGRESSSTATUS_H
#define OP_PROGRESSSTATUS_H
#include "Axomae_macros.h"
#include "Operator.h"

class ProgressStatusWidget;
namespace controller {
  namespace progress_bar {

    struct ProgressBarTextFormat {
      std::string format;
      int percentage;
    };
    template<class T>
    inline int computePercent(T current, T end) {
      ASSERT_IS_ARITHMETIC(T);
      return static_cast<int>(((float)current / (float)end) * 100);
    }
    inline ioperator::OpData<progress_bar::ProgressBarTextFormat> generateData(std::string message, int percentage) {
      progress_bar::ProgressBarTextFormat format{};
      format.format = message;
      format.percentage = percentage;
      ioperator::OpData<progress_bar::ProgressBarTextFormat> data(format);
      return data;
    }
  }  // namespace progress_bar
  class OP_ProgressStatus final : public ioperator::UiOperatorInterface<ProgressStatusWidget, progress_bar::ProgressBarTextFormat> {

   public:
    explicit OP_ProgressStatus(ProgressStatusWidget *progress_bar);
    bool op(ioperator::OpData<progress_bar::ProgressBarTextFormat> *data) const override;
    void reset() const override;
  };

  using ProgressStatus = OP_ProgressStatus;

  /* Use this interface for every object needing to communicate with a progress bar*/
  class IProgressManager {
   public:
    virtual void setProgressManager(ProgressStatus *p_manager) { progress_manager = p_manager; }
    virtual ProgressStatus *getProgressManager() { return progress_manager; }

   protected:
    ProgressStatus *progress_manager{};
  };

}  // namespace controller
#endif
