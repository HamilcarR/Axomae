#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "NovaInterface.h"

#include "ThreadPool.h"
#include "manager/NovaResourceManager.h"
#include "math_camera.h"
#include "ray/Ray.h"
#include "texturing/nova_texturing.h"
#include "utils/nova_utils.h"
#include <vector>

namespace nova {

  class NovaResourceManager;

  /* Local ray engine */
  class NovaRenderEngineLR final : public NovaRenderEngineInterface {
   public:
    CLASS_OCM(NovaRenderEngineLR)

    glm::vec4 engine_sample_color(const Ray &ray, const NovaResourceManager *nova_resources, int depth) override;
    void engine_render_tile(HdrBufferStruct *dest_buffer, Tile &tile, const NovaResourceManager *nova_resources) override;
  };

  inline std::vector<std::future<void>> draw(HdrBufferStruct *buffers,
                                             const unsigned width_resolution,
                                             const unsigned height_resolution,
                                             NovaRenderEngineInterface *engine_instance,
                                             threading::ThreadPool *thread_pool,
                                             const NovaResourceManager *nova_resources) {
    AX_ASSERT(engine_instance != nullptr, "Rendering engine is not initialized.");
    AX_ASSERT(nova_resources != nullptr, "Scene descriptor is not initialized.");
    AX_ASSERT(buffers, "Buffer structure is not initialized.");
    AX_ASSERT(thread_pool != nullptr, "Worker pool is not initialized.");
    std::vector<std::future<void>> futs;
    std::vector<Tile> tiles = divideByTiles(
        width_resolution, height_resolution, nova_resources->getEngineData().getTilesWidth(), nova_resources->getEngineData().getTilesHeight());
    for (auto &elem : tiles) {
      auto renderer_callback = [engine_instance](HdrBufferStruct *buffers, Tile &tile, const NovaResourceManager *resrc) {
        engine_instance->engine_render_tile(buffers, tile, resrc);
      };
      elem.sample_per_tile = nova_resources->getEngineData().getSampleIncrement();
      elem.image_total_height = height_resolution;
      elem.image_total_width = width_resolution;
      futs.push_back(thread_pool->addTask(true, renderer_callback, buffers, elem, nova_resources));
    }
    return futs;
  }

}  // namespace nova
#endif
