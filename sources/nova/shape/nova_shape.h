#ifndef NOVA_SHAPE_H
#define NOVA_SHAPE_H
#include "Axomae_macros.h"
#include "Hitable.h"

#include <memory>
namespace nova {
  class Ray;
}
namespace nova::shape {

  class NovaShapeInterface : public geometry::AABBInterface, public Hitable {
   public:
    ~NovaShapeInterface() override = default;
    [[nodiscard]] virtual glm::vec3 centroid() const = 0;
    template<class SUBTYPE, class... Args>
    static std::unique_ptr<NovaShapeInterface> create(Args &&...args) {
      ASSERT_SUBTYPE(NovaShapeInterface, SUBTYPE);
      return std::make_unique<SUBTYPE>(std::forward<Args>(args)...);
    }
  };
}  // namespace nova::shape

#endif  // NOVA_SHAPE_H
