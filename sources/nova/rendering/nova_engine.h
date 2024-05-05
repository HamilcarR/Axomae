#ifndef NOVA_ENGINE_H
#define NOVA_ENGINE_H

namespace nova::engine {
  struct EngineResourcesHolder {
    int tiles_w{};
    int tiles_h{};
    int sample_increment{};
    int aliasing_samples{};
    int renderer_max_samples{};
  };
}  // namespace nova::engine

#endif
