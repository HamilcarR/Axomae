#ifndef OP_PROGRESSSTATUS_H
#define OP_PROGRESSSTATUS_H
#include "Operator.h"
#include "project_macros.h"
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

  class OperatorProgressStatus final : public ioperator::UiOperatorInterface<ProgressStatusWidget, progress_bar::ProgressBarTextFormat> {

   public:
    explicit OperatorProgressStatus(ProgressStatusWidget *progress_bar);
    bool op(ioperator::OpData<progress_bar::ProgressBarTextFormat> *data) const override;
    void reset() const override;
  };

  using ProgressStatus = OperatorProgressStatus;

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
    void setCurrent(float current_val) { current = current_val; }
    float getTarget() const { return target; }

   protected:
    ProgressStatus *progress_manager{};
    float target;
    float current;
    std::string message;
  };

  /* Provide an object of this class to functions to avoid passing around an IProgressManager "this" pointers*/
  class ProgressManagerHelper {

   public:
    enum PROGRESS_FLAG : unsigned { ZERO = 0, ONE_FOURTH = 1, TWO_FOURTH = 2, THREE_FOURTH = 3, COMPLETE = 4 };
    explicit ProgressManagerHelper(IProgressManager *pm) : progress_manager(pm) {}
    void notifyProgress(float prog);
    void notifyProgress(PROGRESS_FLAG flag);
    void reset();

   private:
    IProgressManager *progress_manager;
  };
}  // namespace controller
#endif
