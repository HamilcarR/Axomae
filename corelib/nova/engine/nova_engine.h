#ifndef NOVA_ENGINE_H
#define NOVA_ENGINE_H
#include "internal/macro/project_macros.h"
#include "utils/nova_utils.h"
#include <atomic>
#include <string>

namespace nova {

  template<class T>
  struct RenderBuffers {
    T *accumulator_buffer;
    T *partial_buffer;
    size_t byte_size_buffers{};
    int channels{};
    T *depth_buffer;
  };
  using HdrBufferStruct = RenderBuffers<float>;

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

     public:
      CLASS_CM(EngineResourcesHolder)
    };
  }  // namespace engine
}  // namespace nova

#endif
