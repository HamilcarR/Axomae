#ifndef NOVA_ENGINE_H
#define NOVA_ENGINE_H

namespace nova::engine {
  struct EngineResourcesHolder {
    int tiles_w{};
    int tiles_h{};
    int render_samples{};
  };
}  // namespace nova::engine

#endif
