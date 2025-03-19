#include "NovaResourceManager.h"
#include "aggregate/acceleration_interface.h"
#include <internal/device/gpgpu/device_transfer_interface.h>
namespace nova {

  void NovaResourceManager::setManagedApiAccelerationStructure(aggregate::DefaultAccelerator &&acceleration_structure) {
    resources.scene_data.api_accelerator = std::move(acceleration_structure);
  }
}  // namespace nova
