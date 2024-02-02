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
      return static_cast<int>((static_cast<float>(current) / static_cast<float>(end)) * 100);
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
    void setProgressManager(ProgressStatus *p_manager) { progress_manager = p_manager; }
    ProgressStatus *getProgressManager() { return progress_manager; }
    void targetProgress(float target_) { target = target_; }
    void notifyProgress();
    void initProgress(ProgressStatus *progress_status, const std::string &message, float target);
    void initProgress(const std::string &message, float target);
    void setProgressStatusText(const std::string &mes) { message = mes; }
    void resetProgress();
    float &getCurrentRefVar() { return current; }

   protected:
    ProgressStatus *progress_manager{};
    float target;
    float current;
    std::string message;
  };

  /* Provide an object of this class to functions to avoid passing around an IProgressManager "this" pointers*/
  class ProgressManagerHelper {
   public:
    explicit ProgressManagerHelper(IProgressManager *pm) : progress_helpee(pm) {}
    void notifyProgress(float prog);

   private:
    IProgressManager *progress_helpee;
  };
}  // namespace controller
#endif
