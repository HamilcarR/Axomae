
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
}  // namespace controller
#endif  // EXCEPTIONHANDLERUI_H
