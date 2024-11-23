#ifndef NOVA_ACCELERATION_H
#define NOVA_ACCELERATION_H
#include "bvh/Bvh.h"

namespace nova::aggregate {
  struct Accelerator {
    Bvhtl accelerator;

    void buildBVH(const axstd::span<primitive::NovaPrimitiveInterface> &primitives,
                  BvhtlBuilder::BUILD_TYPE build_type = BvhtlBuilder::PERFORMANCE,
                  BvhtlBuilder::SEGMENTATION segmentation = BvhtlBuilder::SAH) {
      accelerator.build(primitives, build_type, segmentation);
    }
  };
}  // namespace nova::aggregate
#endif  // NOVA_ACCELERATION_H
