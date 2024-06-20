
#ifndef NOVA_SCENE_H
#define NOVA_SCENE_H
#include "aggregate/bvh/Bvh.h"
#include "nova_primitive.h"
#include "shape/Sphere.h"
#include <memory>

namespace nova::scene {

  // TODO : use memory pool
  struct SceneResourcesHolder {
    std::vector<std::unique_ptr<primitive::NovaPrimitiveInterface>> primitives;
    std::vector<std::unique_ptr<shape::NovaShapeInterface>> shapes;
    std::vector<std::unique_ptr<material::NovaMaterialInterface>> materials_collection;
  };

  struct Accelerator {
    aggregate::Bvhtl accelerator;
  };

}  // namespace nova::scene
#endif  // NOVA_SCENE_H
