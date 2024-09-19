#ifndef NOVAEXCEPTIONMANAGER_H
#define NOVAEXCEPTIONMANAGER_H
#include <engine/nova_exception.h>

namespace nova {
  class NovaExceptionManager {
   private:
    nova::exception::NovaException exception;

   public:
    CLASS_CM(NovaExceptionManager)

    GENERATE_GETTERS(exception::NovaException, ExceptionReference, exception)
    AX_DEVICE_CALLABLE uint64_t checkErrorStatus() const { return exception.errorCheck(); }
    AX_DEVICE_CALLABLE void addError(nova::exception::ERROR error_id) { exception.addErrorType(error_id); }
    AX_DEVICE_CALLABLE void addError(const nova::exception::NovaException &other_exception);
    [[nodiscard]] std::vector<nova::exception::ERROR> getErrorList() const { return exception.getErrorList(); }
  };
}  // namespace nova

#endif  // NOVAEXCEPTIONMANAGER_H
