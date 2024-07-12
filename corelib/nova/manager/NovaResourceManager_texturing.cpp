#include "NovaResourceManager.h"

namespace nova {
  void NovaResourceManager::envmapSetData(std::vector<float> *raw_data, int width, int height, int channels) {
    AX_ASSERT_NOTNULL(raw_data);
    getEnvmapData().raw_data = raw_data;
    getEnvmapData().width = width;
    getEnvmapData().height = height;
    getEnvmapData().channels = channels;
  }
}  // namespace nova