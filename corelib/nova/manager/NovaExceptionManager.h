#ifndef NOVAEXCEPTIONMANAGER_H
#define NOVAEXCEPTIONMANAGER_H
#include <engine/nova_exception.h>

namespace nova {
  class NovaExceptionManager {
   private:
    nova::exception::NovaException exception;

   public:
    ax_device_callable NovaExceptionManager() = default;
    ax_device_callable ~NovaExceptionManager() = default;
    ax_device_callable NovaExceptionManager(const NovaExceptionManager &copy) = default;
    ax_device_callable NovaExceptionManager(NovaExceptionManager &&move) noexcept = default;
    ax_device_callable NovaExceptionManager &operator=(const NovaExceptionManager &copy) = default;
    ax_device_callable NovaExceptionManager &operator=(NovaExceptionManager &&move) noexcept = default;

    GENERATE_GETTERS(exception::NovaException, ExceptionReference, exception)

    ax_device_callable ax_no_discard uint64_t checkErrorStatus() const { return exception.getErrorFlag(); }
    ax_device_callable void addError(uint64_t error_id) { exception.addErrorType(error_id); }
    ax_device_callable void addError(const nova::exception::NovaException &other_exception);
    ax_no_discard std::vector<nova::exception::ERROR> getErrorList() const { return exception.getErrorList(); }
  };
}  // namespace nova

#endif  // NOVAEXCEPTIONMANAGER_H
