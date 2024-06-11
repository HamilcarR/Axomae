#ifndef NOVA_ENGINE_H
#define NOVA_ENGINE_H
#include "nova_utils.h"
#include <atomic>
#include <map>

namespace nova {

  template<class T>
  struct RenderBuffers {
    T *accumulator_buffer;
    T *partial_buffer;
    size_t byte_size_buffers{};
    int channels{};
    std::vector<Tile> tiles;
  };
  using HdrBufferStruct = RenderBuffers<float>;

  namespace engine {
    struct EngineResourcesHolder {
      int tiles_w{};
      int tiles_h{};
      int sample_increment{};
      int aliasing_samples{};
      int renderer_max_samples{};
      int max_depth{};
      std::atomic_long latency;
      bool *cancel_render;
    };
  }  // namespace engine
}  // namespace nova

#endif
