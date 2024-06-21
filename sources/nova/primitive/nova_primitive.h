#ifndef NOVA_PRIMITIVE_H
#define NOVA_PRIMITIVE_H
#include "Axomae_macros.h"
#include "BoundingBox.h"
#include "ray/Hitable.h"
#include <memory>
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
    template<class SUBTYPE, class... Args>
    static std::unique_ptr<NovaPrimitiveInterface> create(Args &&...args) {
      ASSERT_SUBTYPE(NovaPrimitiveInterface, SUBTYPE);
      return std::make_unique<SUBTYPE>(std::forward<Args>(args)...);
    }
  };

}  // namespace nova::primitive
#endif  // NOVA_PRIMITIVE_H
