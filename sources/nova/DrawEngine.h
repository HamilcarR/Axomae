#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "NovaInterface.h"
#include "Ray.h"
#include "ThreadPool.h"
#include "math_camera.h"
#include "nova_utils.h"
#include "texturing/nova_texturing.h"
#include <vector>

namespace nova {

  /* Local ray engine */
  class NovaRenderEngineLR final : public NovaRenderEngineInterface {
   public:
    NovaRenderEngineLR() = default;
    ~NovaRenderEngineLR() override = default;

    glm::vec4 engine_sample_color(const Ray &ray, NovaResources *nova_resources, int depth) override;
    void engine_render_tile(HdrBufferStruct *dest_buffer, Tile &tile, NovaResources *nova_resources) override;
    NovaRenderEngineLR(const NovaRenderEngineLR &copy) = delete;
    NovaRenderEngineLR(NovaRenderEngineLR &&move) noexcept = default;
    NovaRenderEngineLR &operator=(NovaRenderEngineLR &&move) noexcept = default;
    NovaRenderEngineLR &operator=(const NovaRenderEngineLR &copy) = delete;
  };

  inline std::vector<std::future<void>> draw(HdrBufferStruct *buffers,
                                             const unsigned width_resolution,
                                             const unsigned height_resolution,
                                             NovaRenderEngineInterface *engine_instance,
                                             threading::ThreadPool *thread_pool,
                                             NovaResources *nova_resources) {
    AX_ASSERT(engine_instance != nullptr, "Rendering engine is not initialized.");
    AX_ASSERT(nova_resources != nullptr, "Scene descriptor is not initialized.");
    AX_ASSERT(buffers, "Buffer structure is not initialized.");
    AX_ASSERT(thread_pool != nullptr, "Worker pool is not initialized.");
    std::vector<std::future<void>> futs;
    std::vector<Tile> tiles = divideByTiles(
        width_resolution, height_resolution, nova_resources->renderer_data.tiles_w, nova_resources->renderer_data.tiles_h);
    for (auto &elem : tiles) {
      auto renderer_callback = [engine_instance](HdrBufferStruct *buffers, Tile &tile, NovaResources *resrc) {
        engine_instance->engine_render_tile(buffers, tile, resrc);
      };
      elem.sample_per_tile = nova_resources->renderer_data.sample_increment;
      elem.image_total_height = height_resolution;
      elem.image_total_width = width_resolution;
      futs.push_back(thread_pool->addTask(true, renderer_callback, buffers, elem, nova_resources));
    }
    return futs;
  }

}  // namespace nova
#endif
