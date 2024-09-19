#ifndef MANAGERINTERNALSTRUCTS_H
#define MANAGERINTERNALSTRUCTS_H

#include "NovaExceptionManager.h"
#include "NovaResourceManager.h"

namespace nova {
  struct nova_eng_internals {
    const NovaResourceManager *resource_manager;
    NovaExceptionManager *exception_manager;
  };
}  // namespace nova

#endif  // MANAGERINTERNALSTRUCTS_H
