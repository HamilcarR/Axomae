#ifndef PRIVATE_INCLUDES_H
#define PRIVATE_INCLUDES_H
#include "api_camera.h"
#include "api_common.h"
#include "api_engine.h"
#include "api_geometry.h"
#include "api_material.h"
#include "api_renderbuffer.h"
#include "api_renderoptions.h"
#include "api_scene.h"
#include "api_transform.h"

#include "camera/nova_camera.h"
#include "engine/datastructures.h"
#include "manager/NovaResourceManager.h"
#include "material/NovaMaterials.h"
#include "scene/nova_scene.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/math/utils_3D.h>
#include <internal/geometry/Object3D.h>
#include <internal/thread/worker/ThreadPool.h>

constexpr int PBR_TEXTURE_PACK_SIZE = 8;

/* Format is column major */
inline glm::mat4 f16_to_mat4(float arr[16]) {
  glm::mat4 ret;
  for (int i = 0; i < 16; i++)
    glm::value_ptr(ret)[i] = arr[i];
  return ret;
}

inline glm::vec3 f3_to_vec3(float arr[3]) { return {arr[0], arr[1], arr[2]}; }

inline glm::mat4 convert_transform(const float transform[16]) {
  glm::mat4 final_transform;
  for (int i = 0; i < 16; i++) {
    glm::value_ptr(final_transform)[i] = transform[i];
  }
  return final_transform;
}

namespace nova {
  Object3D to_obj3d(const Trimesh &trimesh);
  material::NovaMaterialInterface setup_material_data(const AbstractMesh &mesh, const Material &material, NovaResourceManager &manager);
}  // namespace nova
#endif
