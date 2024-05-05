
#ifndef NOVA_SCENE_H
#define NOVA_SCENE_H
#include "shape/Sphere.h"
namespace nova::scene {
  struct SceneResourcesHolder {
    std::vector<Sphere> objects;
  };
}  // namespace nova::scene
#endif  // NOVA_SCENE_H
