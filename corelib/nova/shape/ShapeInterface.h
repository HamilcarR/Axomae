#ifndef SHAPEINTERFACE_H
#define SHAPEINTERFACE_H
#include "BoundingBox.h"
#include "ray/Hitable.h"
namespace nova::shape {

  class NovaShapeInterface : public geometry::AABBInterface, public Hitable {
   public:
    ~NovaShapeInterface() override = default;
    [[nodiscard]] virtual glm::vec3 centroid() const = 0;
  };
}  // namespace nova::shape
#endif  // SHAPEINTERFACE_H
