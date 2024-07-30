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
    std::vector<NovaShapeInterface> shapes;
    std::vector<std::unique_ptr<Triangle>> tri_allocator;

    template<class SUBTYPE, class... Args>
    NovaShapeInterface add_shape(Args &&...args) {
      static_assert(core::has<SUBTYPE, TYPELIST_SHAPE>::has_type, "Provided type is not a Shape type.");
      if constexpr (ISTYPE(SUBTYPE, Triangle)) {
        tri_allocator.push_back(std::make_unique<SUBTYPE>(std::forward<Args>(args)...));
        shapes.push_back(tri_allocator.back().get());
        AX_ASSERT_NOTNULL(shapes.back().get());
        return shapes.back();
      }
      AX_UNREACHABLE;
      return nullptr;
    }

    std::vector<NovaShapeInterface> &get_shapes() { return shapes; }
    [[nodiscard]] const std::vector<NovaShapeInterface> &get_shapes() const { return shapes; }

    void clear() {
      shapes.clear();
      tri_allocator.clear();
    }
  };

}  // namespace nova::shape

#endif  // NOVA_SHAPE_H
