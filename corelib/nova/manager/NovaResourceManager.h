#ifndef NOVARESOURCEMANAGER_H
#define NOVARESOURCEMANAGER_H
#include "camera/nova_camera.h"
#include "project_macros.h"
#include "ray/Ray.h"
#include "rendering/nova_engine.h"
#include "scene/nova_scene.h"
#include "texturing/nova_texturing.h"

namespace nova {
  class NovaResources {
   public:
    engine::EngineResourcesHolder renderer_data{};
    scene::SceneResourcesHolder scene_data{};

   public:
    CLASS_CM(NovaResources)
  };

  class NovaResourceManager {
   private:
    NovaResources resources;

   public:
    CLASS_CM(NovaResourceManager)

    GENERATE_GETTERS(engine::EngineResourcesHolder, EngineData, resources.renderer_data)
    GENERATE_GETTERS(scene::SceneResourcesHolder, SceneData, resources.scene_data)
    GENERATE_GETTERS(texturing::TextureRawData, EnvmapData, resources.scene_data.envmap_data)
    GENERATE_GETTERS(texturing::TextureResourcesHolder, TexturesData, resources.scene_data.textures_data)
    GENERATE_GETTERS(material::MaterialResourcesHolder, MaterialData, resources.scene_data.materials_data);
    GENERATE_GETTERS(camera::CameraResourcesHolder, CameraData, resources.scene_data.camera_data)
    GENERATE_GETTERS(primitive::PrimitivesResourcesHolder, PrimitiveData, resources.scene_data.primitive_data)
    GENERATE_GETTERS(shape::ShapeResourcesHolder, ShapeData, resources.scene_data.shape_data)
    GENERATE_GETTERS(aggregate::Accelerator, AccelerationData, resources.scene_data.acceleration_data)
    GENERATE_GETTERS(scene::SceneTransformations, SceneTransformation, resources.scene_data.scene_transformations)

    void clearResources() {
      getPrimitiveData().clear();
      getTexturesData().clear();
      getShapeData().clear();
      getMaterialData().clear();
    }

    /* Scene: Textures */
    void envmapSetData(std::vector<float> *raw_data, int width, int height, int channels);
  };
}  // namespace nova

#endif  // NOVARESOURCEMANAGER_H
