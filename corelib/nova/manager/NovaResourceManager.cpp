#include "NovaResourceManager.h"
#include "aggregate/acceleration_interface.h"
#include <internal/device/gpgpu/device_transfer_interface.h>
namespace nova {

  void NovaResourceManager::envmapSetData(float *raw_data, int width, int height, int channels) {
    AX_ASSERT_NOTNULL(raw_data);
    texturing::TextureRawData &envmap_data = getEnvmapData();
    envmap_data.raw_data = raw_data;
    envmap_data.width = width;
    envmap_data.height = height;
    envmap_data.channels = channels;
  }
  void NovaResourceManager::setManagedApiAccelerationStructure(aggregate::DefaultAccelerator &&acceleration_structure) {
    resources.scene_data.api_accelerator = std::move(acceleration_structure);
  }
}  // namespace nova
