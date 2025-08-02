#ifndef MANAGERINTERNALSTRUCTS_H
#define MANAGERINTERNALSTRUCTS_H

namespace nova {

  class NovaResourceManager;
  class NovaExceptionManager;

  struct nova_eng_internals {
    const NovaResourceManager *resource_manager{};
    NovaExceptionManager *exception_manager{};
  };
}  // namespace nova

#endif  // MANAGERINTERNALSTRUCTS_H
