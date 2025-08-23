#include "api_common.h"
#include "private_includes.h"
#include <memory>
namespace nova {

  class NvScene final : public Scene {
    trimesh_group_s trimesh_group;

    std::vector<TexturePtr> envmaps;
    int selected_envmap_id{-1};

    std::vector<CameraPtr> cameras;
    int selected_camera_id{-1};

    TransformPtr transform;  // root transform

   public:
    ERROR_STATE addMesh(TrimeshPtr mesh, MaterialPtr material) override {
      trimesh_group.mesh_geometry.emplace_back(std::move(mesh));
      trimesh_group.mesh_material.emplace_back(std::move(material));
      return SUCCESS;
    }

    ERROR_STATE useEnvmap(unsigned id) override {
      if (id >= envmaps.size())
        return INVALID_ARGUMENT;
      selected_envmap_id = (unsigned)id;
      return SUCCESS;
    }

    int getCurrentEnvmapId() const override { return selected_envmap_id; }

    Texture *getEnvmap(unsigned id) override { return envmaps.size() <= id ? nullptr : envmaps[id].get(); }

    const Texture *getEnvmap(unsigned id) const override { return envmaps.size() <= id ? nullptr : envmaps[id].get(); }

    unsigned addEnvmap(TexturePtr envmap_texture) override {
      envmaps.push_back(std::move(envmap_texture));
      if (selected_envmap_id == -1)
        selected_envmap_id = 0;
      return envmaps.size() - 1;
    }

    ERROR_STATE useCamera(unsigned id) override {
      if (id >= cameras.size())
        return INVALID_ARGUMENT;
      selected_camera_id = id;
      return SUCCESS;
    }

    Camera *getCamera(unsigned id) override { return cameras.size() <= id ? nullptr : cameras[id].get(); }

    const Camera *getCamera(unsigned id) const override { return cameras.size() <= id ? nullptr : cameras[id].get(); }

    unsigned addCamera(CameraPtr camera) override {
      cameras.push_back(std::move(camera));
      if (selected_camera_id == -1)
        selected_camera_id = 0;
      return cameras.size() - 1;
    }

    int getCurrentCameraId() const override { return selected_camera_id; }

    ERROR_STATE setRootTransform(TransformPtr t) override {
      transform = std::move(t);
      return SUCCESS;
    }

    unsigned getPrimitivesNum(mesh::TYPE type) const override {
      switch (type) {
        case mesh::TRIANGLE:
        default:
          return getTrianglePrimitivesNum();
          break;
      }
    }

    unsigned getMeshesNum(mesh::TYPE type) const override {
      switch (type) {
        case mesh::TRIANGLE:
        default:
          return trimesh_group.mesh_geometry.size();
          break;
      }
    }

    unsigned getTrianglePrimitivesNum() const {
      std::size_t acc = 0;
      const int trimesh_index_padding = 3;
      for (const auto &elem : trimesh_group.mesh_geometry) {
        const Object3D &mesh_geometry = to_obj3d(*elem);
        AX_ASSERT_NOTNULL(elem);
        for (std::size_t i = 0; i < mesh_geometry.indices.size(); i += trimesh_index_padding)
          acc++;
      }
      return acc;
    }

    CsteTriMeshCollection getTriangleMeshCollection() const override { return trimesh_group.mesh_geometry; }

    CsteMaterialCollection getMaterialCollection(mesh::TYPE type) const override {
      switch (type) {
        case mesh::TRIANGLE:
        default:
          return trimesh_group.mesh_material;
          break;
      }
    }

    CsteCameraCollection getCameraCollection() const override { return cameras; }

    CsteTextureCollection getEnvmapCollection() const override { return envmaps; }
  };

  std::unique_ptr<Scene> create_scene() { return std::make_unique<NvScene>(); }

}  // namespace nova
