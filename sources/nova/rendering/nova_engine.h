#ifndef NOVA_ENGINE_H
#define NOVA_ENGINE_H
#include <map>

namespace nova {

  template<class T>
  struct RenderBuffers {
    T *accumulator_buffer;
    T *partial_buffer;
    size_t byte_size_buffers;
    int channels;
  };
  using HdrBufferStruct = RenderBuffers<float>;

  namespace engine {
    struct EngineResourcesHolder {
      int tiles_w{};
      int tiles_h{};
      int sample_increment{};
      int aliasing_samples{};
      int renderer_max_samples{};
    };
  }  // namespace engine
}  // namespace nova

#endif
