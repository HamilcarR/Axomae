#ifndef NOVAEXCEPTIONMANAGER_H
#define NOVAEXCEPTIONMANAGER_H
#include <engine/nova_exception.h>

namespace nova {
  class NovaExceptionManager {
   private:
    nova::exception::NovaException exception;

   public:
    AX_DEVICE_CALLABLE NovaExceptionManager() = default;
    AX_DEVICE_CALLABLE ~NovaExceptionManager() = default;
    AX_DEVICE_CALLABLE NovaExceptionManager(const NovaExceptionManager &copy) = default;
    AX_DEVICE_CALLABLE NovaExceptionManager(NovaExceptionManager &&move) noexcept = default;
    AX_DEVICE_CALLABLE NovaExceptionManager &operator=(const NovaExceptionManager &copy) = default;
    AX_DEVICE_CALLABLE NovaExceptionManager &operator=(NovaExceptionManager &&move) noexcept = default;

    GENERATE_GETTERS(exception::NovaException, ExceptionReference, exception)

    AX_DEVICE_CALLABLE ax_no_discard uint64_t checkErrorStatus() const { return exception.getErrorFlag(); }
    AX_DEVICE_CALLABLE void addError(uint64_t error_id) { exception.addErrorType(error_id); }
    AX_DEVICE_CALLABLE void addError(const nova::exception::NovaException &other_exception);
    ax_no_discard std::vector<nova::exception::ERROR> getErrorList() const { return exception.getErrorList(); }
  };
}  // namespace nova

#endif  // NOVAEXCEPTIONMANAGER_H
