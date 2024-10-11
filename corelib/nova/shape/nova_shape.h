#ifndef NOVA_SHAPE_H
#define NOVA_SHAPE_H
#include "Triangle.h"
#include "internal/memory/MemoryArena.h"
#include <memory>

namespace nova {
  class Ray;
}
namespace nova::shape {

  struct ShapeResourcesHolder {
    std::vector<NovaShapeInterface> shapes;

    /* Since the creation of shapes is incremental (we don't copy a whole array here) , provide the current offset of the object to create.*/
    template<class T, class... Args>
    NovaShapeInterface add_shape(T *allocation_buffer, std::size_t offset, Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Shape type.");
      T *allocated_ptr = core::memory::MemoryArena<>::construct<T>(&allocation_buffer[offset], std::forward<Args>(args)...);
      shapes.push_back(allocated_ptr);
      AX_ASSERT_NOTNULL(shapes.back().get());
      return shapes.back();
    }

    std::vector<NovaShapeInterface> &get_shapes() { return shapes; }
    [[nodiscard]] const std::vector<NovaShapeInterface> &get_shapes() const { return shapes; }

    void clear() { shapes.clear(); }
  };

}  // namespace nova::shape

#endif  // NOVA_SHAPE_H
