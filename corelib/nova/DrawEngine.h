#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "NovaInterface.h"
#include "nova_gpu_utils.h"

#include <future>

namespace threading {
  class ThreadPool;
}
namespace nova {

  class NovaRenderEngineLR final : public NovaRenderEngineInterface {
   public:
    CLASS_OCM(NovaRenderEngineLR)

    void engine_render_tile(HdrBufferStruct *dest_buffer, Tile &tile, nova::nova_eng_internals &nova_internals) override;
  };

  std::vector<std::future<void>> draw(HdrBufferStruct *buffers,
                                      unsigned width_resolution,
                                      unsigned height_resolution,
                                      NovaRenderEngineInterface *engine_instance,
                                      threading::ThreadPool *thread_pool,
                                      nova::nova_eng_internals &nova_internals);

  void gpu_draw(HdrBufferStruct *buffers,
                unsigned width_resolution,
                unsigned height_resolution,
                nova::nova_eng_internals &nova_internals,
                gputils::gpu_util_structures_t &gpu_structures);

}  // namespace nova
#endif
