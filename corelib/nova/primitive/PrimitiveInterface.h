
#ifndef PRIMITIVEINTERFACE_H
#define PRIMITIVEINTERFACE_H
#include "ray/Hitable.h"

namespace nova::shape {
  class NovaShapeInterface;
}
namespace nova::material {
  class NovaMaterialInterface;
}
namespace nova::primitive {
  class NovaPrimitiveInterface : public Hitable, public geometry::AABBInterface {
   public:
    ~NovaPrimitiveInterface() override = default;
    virtual bool scatter(const Ray &in, Ray &out, hit_data &data) const = 0;
    [[nodiscard]] virtual glm::vec3 centroid() const = 0;
  };

}  // namespace nova::primitive

#endif  // PRIMITIVEINTERFACE_H
