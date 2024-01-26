#ifndef ILOCKABLE_H
#define ILOCKABLE_H
#include "Mutex.h"

class ILockable {

 protected:
  mutable Mutex mutex;
};

#endif  // ILOCKABLE_H
