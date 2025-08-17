#include "api_common.h"
#include "private_includes.h"
#include <memory>
namespace nova {
  ERROR_STATE NvScene::addMesh(const TriMesh &mesh, const Material &material) {
    trimesh_object_s trimesh;
    trimesh.mesh_geometry = std::make_unique<NvTriMesh>();
    trimesh.mesh_material = std::make_unique<NvMaterial>();
    trimesh_group.emplace_back(std::move(trimesh));
    return SUCCESS;
  }

  ERROR_STATE NvScene::addEnvmap(const Texture &envmap_texture) {
    envmaps.push_back(envmap_texture);
    return SUCCESS;
  }

  ERROR_STATE NvScene::addCamera(const Camera &camera) {}

  ERROR_STATE NvScene::addRootTransform(const float transform[16]) {}

  axstd::span<const trimesh_object_s> NvScene::getTrimeshArray() const { return trimesh_group; }

  std::unique_ptr<Scene> create_scene() { return std::make_unique<NvScene>(); }

}  // namespace nova
