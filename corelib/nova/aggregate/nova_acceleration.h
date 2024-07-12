#ifndef NOVA_ACCELERATION_H
#define NOVA_ACCELERATION_H
#include "bvh/Bvh.h"

namespace nova::aggregate {
  struct Accelerator {
    Bvhtl accelerator;

    template<class... Args>
    void build(Args &&...args) {
      accelerator.build(std::forward<Args>(args)...);
    }
  };
}  // namespace nova::aggregate
#endif  // NOVA_ACCELERATION_H
