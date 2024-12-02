#ifndef NOVA_MATERIAL_H
#define NOVA_MATERIAL_H
#include "NovaMaterials.h"
#include "internal/memory/MemoryArena.h"
#include <memory>
namespace nova::material {

  struct MaterialResourcesHolder {
    std::vector<NovaMaterialInterface> materials;

    template<class T, class... Args>
    NovaMaterialInterface add_material(core::memory::ByteArena &arena, T *allocation_buffer, std::size_t offset, Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Material type.");
      T *allocated_ptr = arena.constructAtMemPosition(allocation_buffer, offset, std::forward<Args>(args)...);
      materials.push_back(allocated_ptr);
      return materials.back();
    }
    std::vector<NovaMaterialInterface> &get_materials() { return materials; }
    ax_no_discard const std::vector<NovaMaterialInterface> &get_materials() const { return materials; }

    void clear() { materials.clear(); }
  };

}  // namespace nova::material

#endif
