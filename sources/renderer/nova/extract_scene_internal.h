#ifndef EXTRACT_SCENE_INTERNAL_H
#define EXTRACT_SCENE_INTERNAL_H
#include "EnvmapTextureManager.h"
#include "bake_render_data.h"

namespace nova_baker_utils {

  void setup_mesh(const drawable_original_transform &drawable, nova::Trimesh &mesh);
  void setup_material(const drawable_original_transform &drawable, nova::Material &material);
}  // namespace nova_baker_utils
#endif
