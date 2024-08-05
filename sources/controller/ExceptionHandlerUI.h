
#ifndef EXCEPTIONHANDLERUI_H
#define EXCEPTIONHANDLERUI_H
#include "GenericException.h"
#include <cstdlib>
#include <string>

namespace controller {

  class ExceptionHandlerUI {
   public:
    static int launchInfoBox(const std::string &message, exception::SEVERITY severity = exception::INFO);
    static exception::SEVERITY intToSeverity(int n);
  };

  class ExceptionInfoBoxHandler {
   public:
    static void handle(const std::string &message, exception::SEVERITY severity) { controller::ExceptionHandlerUI::launchInfoBox(message, severity); }
    static void handle(const exception::ExceptionData &e) {
      exception::SEVERITY s = controller::ExceptionHandlerUI::intToSeverity(e.getSeverity());
      controller::ExceptionHandlerUI::launchInfoBox(e.getErrorMessage(), s);
    }
  };

}  // namespace controller
#endif  // EXCEPTIONHANDLERUI_H
