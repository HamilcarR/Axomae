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

  struct device_traversal_param_s;
  struct nova_eng_internals;

  void gpu_draw(const device_traversal_param_s &traversal_params, nova_eng_internals &internals);

}  // namespace nova
#endif
