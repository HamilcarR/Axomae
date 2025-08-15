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
}  // namespace nova_baker_utils
#endif
