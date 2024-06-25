#ifndef NOVA_PRIMITIVE_H
#define NOVA_PRIMITIVE_H
#include "Axomae_macros.h"
#include "BoundingBox.h"
#include "NovaGeoPrimitive.h"
#include "ray/Hitable.h"
#include "utils/macros.h"
#include <memory>

namespace nova::primitive {
  struct PrimitivesResourcesHolder {
    std::vector<std::unique_ptr<NovaPrimitiveInterface>> primitives;

    REGISTER_RESOURCE(primitive, NovaPrimitiveInterface, primitives)
  };

  RESOURCES_DEFINE_CREATE(NovaPrimitiveInterface)

}  // namespace nova::primitive
#endif  // NOVA_PRIMITIVE_H
