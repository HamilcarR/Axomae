#ifndef NOVARESOURCEMANAGER_H
#define NOVARESOURCEMANAGER_H
#include "aggregate/acceleration_interface.h"
#include "camera/nova_camera.h"
#include "engine/nova_engine.h"
#include "gpu/nova_gpu.h"
#include "scene/nova_scene.h"
#include "texturing/nova_texturing.h"
#include <internal/macro/class_macros.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/MemoryArena.h>

namespace nova {

  class NovaResources {
   public:
    engine::EngineResourcesHolder renderer_data{};
    scene::SceneResourcesHolder scene_data{};

   public:
    CLASS_M(NovaResources)
  };

  namespace aggregate {
    class DeviceAcceleratorInterface;
  }

  class NovaResourceManager {
   private:
    NovaResources resources;

    /* Holds the actual objects refered to by the resource structures.*/
    axstd::ByteArena resource_mempool;

   public:
    NovaResourceManager() = default;
    ~NovaResourceManager();
    NovaResourceManager(NovaResourceManager &&) noexcept;
    NovaResourceManager &operator=(NovaResourceManager &&) noexcept;

    NovaResourceManager(const NovaResourceManager &) = delete;
    NovaResourceManager &operator=(const NovaResourceManager &) = delete;

    engine::EngineResourcesHolder &getEngineData() { return resources.renderer_data; }
    const engine::EngineResourcesHolder &getEngineData() const { return resources.renderer_data; }

    scene::SceneResourcesHolder &getSceneData() { return resources.scene_data; }
    const scene::SceneResourcesHolder &getSceneData() const { return resources.scene_data; }

    texturing::TextureResourcesHolder &getTexturesData() { return resources.scene_data.textures_data; }
    const texturing ::TextureResourcesHolder &getTexturesData() const { return resources.scene_data.textures_data; }

    material::MaterialResourcesHolder &getMaterialData() { return resources.scene_data.materials_data; }
    const material::MaterialResourcesHolder &getMaterialData() const { return resources.scene_data.materials_data; }

    camera::CameraResourcesHolder &getCameraData() { return resources.scene_data.camera_data; }
    const camera::CameraResourcesHolder &getCameraData() const { return resources.scene_data.camera_data; }

    primitive::PrimitivesResourcesHolder &getPrimitiveData() { return resources.scene_data.primitive_data; }
    const primitive::PrimitivesResourcesHolder &getPrimitiveData() const { return resources.scene_data.primitive_data; }

    shape::ShapeResourcesHolder &getShapeData() { return resources.scene_data.shape_data; }
    const shape::ShapeResourcesHolder &getShapeData() const { return resources.scene_data.shape_data; }

    scene::SceneTransformations &getSceneTransformation() { return resources.scene_data.scene_transformations; }
    const scene::SceneTransformations &getSceneTransformation() const { return resources.scene_data.scene_transformations; }

    axstd::MemoryArena<> &getMemoryPool() { return resource_mempool; }
    const axstd::MemoryArena<> &getMemoryPool() const { return resource_mempool; }

    /* Will take ownership of acceleration_structure */
    void setManagedCpuAccelerationStructure(aggregate::DefaultAccelerator &&acceleration_structure);
    aggregate::DefaultAccelerator &getCpuManagedAccelerator() { return resources.scene_data.api_accelerator; }
    const aggregate::DefaultAccelerator &getCpuManagedAccelerator() const { return resources.scene_data.api_accelerator; }

    void setManagedGpuAccelerationStructure(std::unique_ptr<aggregate::DeviceAcceleratorInterface> acceleration_structure);
    aggregate::DeviceAcceleratorInterface *getGpuManagedAccelerator() { return resources.scene_data.device_accelerator.get(); }
    const aggregate::DeviceAcceleratorInterface *getGpuManagedAccelerator() const { return resources.scene_data.device_accelerator.get(); }
    void registerDeviceParameters(const device_traversal_param_s &params) const;

    void clearResources();
  };
}  // namespace nova

#endif  // NOVARESOURCEMANAGER_H
