#include "NovaExceptionManager.h"
namespace nova {
  void NovaExceptionManager::addError(const nova::exception::NovaException &other_exception) {
    auto other_flag = other_exception.getErrorFlag();
    exception.merge(other_flag);
  }
}  // namespace nova
