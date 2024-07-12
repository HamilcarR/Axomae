#ifndef NOVA_SHAPE_H
#define NOVA_SHAPE_H
#include "Box.h"
#include "Sphere.h"
#include "Square.h"
#include "Triangle.h"
#include "utils/macros.h"
#include <memory>
namespace nova {
  class Ray;
}
namespace nova::shape {

  struct ShapeResourcesHolder {
    std::vector<std::unique_ptr<NovaShapeInterface>> shapes;

    REGISTER_RESOURCE(shape, NovaShapeInterface, shapes)
  };

  RESOURCES_DEFINE_CREATE(NovaShapeInterface)

}  // namespace nova::shape

#endif  // NOVA_SHAPE_H
