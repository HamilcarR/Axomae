#ifndef NOVA_SHAPE_H
#define NOVA_SHAPE_H
#include "Axomae_macros.h"
#include <memory>
namespace nova {
  class Ray;
}
namespace nova::shape {
  class NovaShapeInterface : public geometry::AABBInterface {
   public:
    ~NovaShapeInterface() override = default;
    virtual bool intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal_at_intersection, float &t) const = 0;
    [[nodiscard]] virtual glm::vec3 centroid() const = 0;
    template<class SUBTYPE, class... Args>
    static std::unique_ptr<SUBTYPE> create(Args &&...args) {
      ASSERT_SUBTYPE(NovaShapeInterface, SUBTYPE);
      return std::make_unique<SUBTYPE>(std::forward<Args>(args)...);
    }
  };
}  // namespace nova::shape

#endif  // NOVA_SHAPE_H
