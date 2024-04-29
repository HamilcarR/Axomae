#ifndef NOVA_DRAW_ENGINE_H
#define NOVA_DRAW_ENGINE_H
#include "Hitable.h"
#include "NovaInterface.h"
#include "Ray.h"
#include "ThreadPool.h"
#include "nova_texturing.h"
#include "nova_utils.h"
#include <vector>

namespace nova {

  class NovaRenderEngineLR : public NovaRenderEngineInterface<NovaRenderEngineLR> {
   public:
    NovaRenderEngineLR() = default;
    virtual ~NovaRenderEngineLR() = default;

    glm::vec4 engine_sample_color(const Ray &ray, const NovaResources *nova_resources);
    void engine_render_tile(float *dest_buffer,
                            int width_limit_low,
                            int width_limit_high,
                            int height_limit_low,
                            int height_limit_high,
                            const NovaResources *nova_resources);
    NovaRenderEngineLR(const NovaRenderEngineLR &copy) = delete;
    NovaRenderEngineLR(NovaRenderEngineLR &&move) noexcept = default;
    NovaRenderEngineLR &operator=(NovaRenderEngineLR &&move) noexcept = default;
    NovaRenderEngineLR &operator=(const NovaRenderEngineLR &copy) = delete;
  };

  template<class R>
  inline std::vector<std::future<void>> draw(float *display_buffer,
                                             const unsigned width_resolution,
                                             const unsigned height_resolution,
                                             NovaRenderEngineInterface<R> *engine_instance,
                                             threading::ThreadPool *thread_pool,
                                             const NovaResources *nova_resources) {
    AX_ASSERT(engine_instance != nullptr, "Rendering engine is null");
    AX_ASSERT(nova_resources != nullptr, "Scene descriptor is null");
    AX_ASSERT(thread_pool != nullptr, "");
    std::vector<std::future<void>> futs;
    int THREAD_NUM = thread_pool->threadNumber();
    std::vector<Tile> tiles = divideByTiles(width_resolution, height_resolution, THREAD_NUM);
    for (const auto &elem : tiles) {
      auto renderer_callback =
          [&engine_instance](
              float *display_buffer, int width_start, int width_end, int height_start, int height_end, const NovaResources *nova_resources) {
            engine_instance->render_tile(display_buffer, width_start, width_end, height_start, height_end, nova_resources);
          };
      futs.push_back(thread_pool->addTask(
          true, renderer_callback, display_buffer, elem.width_start, elem.width_end, elem.height_start, elem.height_end, nova_resources));
    }
    return futs;
  }

}  // namespace nova
#endif
