#ifndef NOVA_ENGINE_H
#define NOVA_ENGINE_H
#include "datastructures.h"
#include "gpu/nova_gpu.h"
#include "utils/nova_utils.h"
#include <atomic>
#include <internal/macro/project_macros.h>
#include <string>
namespace nova {

  namespace engine {
    class EngineResourcesHolder {
     public:
      int tiles_width{};
      int tiles_height{};
      int sample_increment{};
      int aliasing_samples{};
      int renderer_max_samples{};
      int max_depth{};
      bool is_rendering{};
      bool vertical_invert{false};
      std::string threadpool_tag;
      int integrator_flag{};

      CLASS_CM(EngineResourcesHolder)
    };
  }  // namespace engine
}  // namespace nova

#endif
