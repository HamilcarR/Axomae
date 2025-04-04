#ifndef NOVA_PRIMITIVE_H
#define NOVA_PRIMITIVE_H
#include "PrimitiveInterface.h"
#include "internal/geometry/BoundingBox.h"
#include "internal/memory/MemoryArena.h"

namespace nova::primitive {
  struct PrimitivesResourcesHolder {
    std::vector<NovaPrimitiveInterface> primitives;

    template<class T, class... Args>
    NovaPrimitiveInterface add_primitive(T *allocation_buffer, std::size_t offset, Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Primitive type.");
      T *allocated_ptr = core::memory::MemoryArena<>::construct<T>(&allocation_buffer[offset], std::forward<Args>(args)...);
      primitives.push_back(allocated_ptr);
      return primitives.back();
    }

    std::vector<NovaPrimitiveInterface> &get_primitives() { return primitives; }
    ax_no_discard const std::vector<NovaPrimitiveInterface> &get_primitives() const { return primitives; }

    primitives_view_tn getView() { return {primitives.data(), primitives.size()}; }

    void clear() { primitives.clear(); }
  };

}  // namespace nova::primitive
#endif  // NOVA_PRIMITIVE_H
