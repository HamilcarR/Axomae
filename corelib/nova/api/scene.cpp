#include "private_includes.h"
#include <memory>
namespace nova {
  ERROR_STATE NvScene::addMesh(const NvAbstractTriMesh &mesh, const NvAbstractMaterial &material) {
    trimesh_object_s trimesh;
    trimesh.mesh_geometry = std::make_unique<NvTriMesh>();
    trimesh.mesh_material = std::make_unique<NvMaterial>();
    trimesh_group.emplace_back(trimesh);
    return SUCCESS;
  }

  std::unique_ptr<NvAbstractScene> create_scene() { return std::make_unique<NvScene>(); }
}  // namespace nova
