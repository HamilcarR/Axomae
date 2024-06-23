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

  struct ShapeResourceHolder {
    std::vector<std::unique_ptr<NovaShapeInterface>> shapes;

    RESOURCES_DEFINE_ADD(shape, NovaShapeInterface, shapes)
  };

  RESOURCES_DEFINE_CREATE(NovaShapeInterface)

}  // namespace nova::shape

#endif  // NOVA_SHAPE_H
