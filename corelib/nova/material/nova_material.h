#ifndef NOVA_MATERIAL_H
#define NOVA_MATERIAL_H
#include "MemoryArena.h"
#include "NovaMaterials.h"
#include "project_macros.h"
#include "utils/macros.h"
#include <memory>
namespace nova::material {

  struct MaterialResourcesHolder {
    std::vector<NovaMaterialInterface> materials;

    template<class T, class... Args>
    NovaMaterialInterface add_material(T *allocation_buffer, std::size_t offset, Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Material type.");
      T *allocated_ptr = core::memory::Arena<>::construct<T>(&allocation_buffer[offset], std::forward<Args>(args)...);
      materials.push_back(allocated_ptr);
      AX_ASSERT_NOTNULL(allocation_buffer[offset].get());
      return materials.back();
    }
    std::vector<NovaMaterialInterface> &get_materials() { return materials; }
    [[nodiscard]] const std::vector<NovaMaterialInterface> &get_materials() const { return materials; }

    void clear() { materials.clear(); }
  };

}  // namespace nova::material

#endif
