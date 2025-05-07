#ifndef EXTRACT_SCENE_INTERNAL_H
#define EXTRACT_SCENE_INTERNAL_H
#include "EnvmapTextureManager.h"
#include "bake_render_data.h"

namespace nova_baker_utils {

  void setup_geometry_data(const drawable_original_transform &drawable,
                           nova::material::NovaMaterialInterface &material,
                           nova::NovaResourceManager &manager,
                           std::size_t mesh_index);
  nova::material::NovaMaterialInterface setup_material_data(const Drawable &drawable, nova::NovaResourceManager &manager);
  primitive_buffers_t allocate_primitive_triangle_buffers(core::memory::ByteArena &memory_pool, std::size_t number_elements);
  void extract_envmap(const EnvmapTextureManager &envmap_manager, nova::NovaResourceManager &nova_manager);
}  // namespace nova_baker_utils
#endif
