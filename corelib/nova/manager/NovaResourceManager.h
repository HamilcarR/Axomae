#ifndef NOVARESOURCEMANAGER_H
#define NOVARESOURCEMANAGER_H
#include "aggregate/acceleration_interface.h"
#include "camera/nova_camera.h"
#include "engine/nova_engine.h"
#include "engine/nova_exception.h"
#include "ray/Ray.h"
#include "scene/nova_scene.h"
#include "texturing/nova_texturing.h"
#include <internal/macro/class_macros.h>
#include <internal/macro/project_macros.h>

namespace nova {
  class NovaResources {
   public:
    engine::EngineResourcesHolder renderer_data{};
    scene::SceneResourcesHolder scene_data{};

   public:
    CLASS_M(NovaResources)
  };

  class NovaResourceManager {
   private:
    NovaResources resources;

    /* Holds the actual objects refered to by the resource structures.*/
    core::memory::ByteArena resource_mempool;

   public:
    NovaResourceManager() = default;
    ~NovaResourceManager() = default;
    NovaResourceManager(const NovaResourceManager &) = delete;
    NovaResourceManager(NovaResourceManager &&) noexcept = default;
    NovaResourceManager &operator=(const NovaResourceManager &) = delete;
    NovaResourceManager &operator=(NovaResourceManager &&) noexcept = default;

    GENERATE_GETTERS(engine::EngineResourcesHolder, EngineData, resources.renderer_data)
    GENERATE_GETTERS(scene::SceneResourcesHolder, SceneData, resources.scene_data)
    GENERATE_GETTERS(texturing::TextureResourcesHolder, TexturesData, resources.scene_data.textures_data)
    GENERATE_GETTERS(material::MaterialResourcesHolder, MaterialData, resources.scene_data.materials_data)
    GENERATE_GETTERS(camera::CameraResourcesHolder, CameraData, resources.scene_data.camera_data)
    GENERATE_GETTERS(primitive::PrimitivesResourcesHolder, PrimitiveData, resources.scene_data.primitive_data)
    GENERATE_GETTERS(shape::ShapeResourcesHolder, ShapeData, resources.scene_data.shape_data)
    GENERATE_GETTERS(aggregate::DefaultAccelerator, APIManagedAccelerator, resources.scene_data.api_accelerator)
    GENERATE_GETTERS(scene::SceneTransformations, SceneTransformation, resources.scene_data.scene_transformations)
    GENERATE_GETTERS(core::memory::MemoryArena<>, MemoryPool, resource_mempool)

    void clearResources() {
      resource_mempool.reset();
      getPrimitiveData().clear();
      getTexturesData().clear();
      getShapeData().clear();
      getMaterialData().clear();
      getAPIManagedAccelerator().cleanup();
    }

    /* Will take ownership of acceleration_structure */
    void setManagedApiAccelerationStructure(aggregate::DefaultAccelerator &&acceleration_structure);
  };
}  // namespace nova

#endif  // NOVARESOURCEMANAGER_H
