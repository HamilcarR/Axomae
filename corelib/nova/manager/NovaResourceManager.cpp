#include "NovaResourceManager.h"
#include "aggregate/acceleration_interface.h"
#include "aggregate/device_acceleration_interface.h"
namespace nova {

  NovaResourceManager::~NovaResourceManager() = default;

  NovaResourceManager::NovaResourceManager(NovaResourceManager &&) noexcept = default;

  NovaResourceManager &NovaResourceManager::operator=(NovaResourceManager &&) noexcept = default;

  void NovaResourceManager::setManagedCpuAccelerationStructure(aggregate::DefaultAccelerator &&acceleration_structure) {
    resources.scene_data.api_accelerator = std::move(acceleration_structure);
  }

  void NovaResourceManager::setManagedGpuAccelerationStructure(std::unique_ptr<aggregate::DeviceAcceleratorInterface> acceleration_structure) {
    resources.scene_data.device_accelerator = std::move(acceleration_structure);
  }

  void NovaResourceManager::clearResources() {
    resource_mempool.reset();
    getPrimitiveData().clear();
    getTexturesData().clear();
    getShapeData().clear();
    getMaterialData().clear();
    getCpuManagedAccelerator().cleanup();
#ifdef AXOMAE_USE_CUDA
    if (getGpuManagedAccelerator())
      getGpuManagedAccelerator()->cleanup();
#endif
  }
}  // namespace nova
