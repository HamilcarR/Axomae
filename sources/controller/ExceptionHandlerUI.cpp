#include "ExceptionHandlerUI.h"
#include "MessageBoxWidget.h"

namespace controller {

  int ExceptionHandlerUI::launchInfoBox(const std::string &message, exception::SEVERITY severity) {
    switch (severity) {
      case exception::WARNING: {
        WarningBoxWidget warn(message);
        return warn.exec();
      }
      case exception::INFO: {
        InfoBoxWidget info(message);
        return info.exec();
      }
      case exception::CRITICAL: {
        CriticalBoxWidget crit(message);
        return crit.exec();
      }
      default: {
        WarningBoxWidget def(message);
        return def.exec();
      }
    }
  }
  exception::SEVERITY ExceptionHandlerUI::intToSeverity(int n) {
    if (n == 0)
      return exception::CRITICAL;
    if (n == 1)
      return exception::WARNING;
    if (n == 2)
      return exception::INFO;
    else
      return exception::WARNING;
  }

}  // namespace controller