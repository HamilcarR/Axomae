#ifndef NOVA_ACCELERATION_H
#define NOVA_ACCELERATION_H
#include "bvh/Bvh.h"

namespace nova::aggregate {
  struct Accelerator {
    Bvhtl accelerator;
  };
}  // namespace nova::aggregate
#endif  // NOVA_ACCELERATION_H
