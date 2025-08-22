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

  unsigned NvScene::addEnvmap(const Texture &envmap_texture) {
    envmaps.push_back(envmap_texture);
    return envmaps.size() - 1;
  }

  unsigned NvScene::addCamera(const Camera &camera) {
    cameras.push_back(camera);
    return cameras.size() - 1;
  }

  ERROR_STATE NvScene::setRootTransform(const Transform &t) {
    transform = t;
    return SUCCESS;
  }

  axstd::span<const trimesh_object_s> NvScene::getTrimeshArray() const { return trimesh_group; }

  std::unique_ptr<Scene> create_scene() { return std::make_unique<NvScene>(); }

}  // namespace nova
