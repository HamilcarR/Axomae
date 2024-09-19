#include "DrawEngine.h"
#include "GenericException.h"
#include "Logger.h"
#include "engine/nova_engine.h"
#include "integrator/Integrator.h"
#include "manager/NovaResourceManager.h"

namespace exception {
  class InvalidThreadpoolStateException final : public GenericException {
   public:
    explicit InvalidThreadpoolStateException(const std::string &err) : GenericException() { saveErrorString(err); }
  };

  class InvalidResourceManagerStateException final : public GenericException {
   public:
    explicit InvalidResourceManagerStateException(const std::string &err) : GenericException() { saveErrorString(err); }
  };
}  // namespace exception

namespace nova {
  /**
   * TODO : Will use different states :
   * 1) Fast state : On move events , on redraw , on resize etc will trigger fast state .
   * The scheduler needs to be emptied , threads synchronized and stopped , and we redraw with
   * 1 ray/pixel , at 1 sample , at 1 depth in each tile , then copy the sampled value to the other pixels.
   * 2) Intermediary state : increase logarithmically the amount of pixels sampled + number of samples , at half depth .
   * 3) Final state : render at full depth , full sample size , full resolution.
   * Allows proper synchronization.
   */
  void NovaRenderEngineLR::engine_render_tile(HdrBufferStruct *buffers, Tile &tile, nova::nova_eng_internals &nova_internals) {
    integrator::integrator_dispatch(buffers, tile, nova_internals);
  }

  static bool validate(HdrBufferStruct *buffers,
                       const unsigned width_resolution,
                       const unsigned height_resolution,
                       NovaRenderEngineInterface *engine_instance,
                       threading::ThreadPool *thread_pool,
                       nova::nova_eng_internals &nova_internals) {

    bool keep_rendering = true;
    const NovaResourceManager *nova_resources = nova_internals.resource_manager;
    NovaExceptionManager *nova_exception = nova_internals.exception_manager;
    if (!nova_resources)
      throw ::exception::InvalidResourceManagerStateException("Resource manager reference is invalid (nullptr).");
    if (!nova_exception)
      throw ::exception::InvalidResourceManagerStateException("Exception manager reference is invalid (nullptr).");
    if (!engine_instance) {
      nova_exception->addError(nova::exception::INVALID_ENGINE_INSTANCE);
      keep_rendering = false;
    }
    if (width_resolution == 0 || height_resolution == 0 || width_resolution > 65536 || height_resolution > 65536) {
      nova_exception->addError(nova::exception::INVALID_RENDERBUFFER_DIM);
      keep_rendering = false;
    }
    if (!buffers) {
      nova_exception->addError(nova::exception::INVALID_RENDERBUFFER_STATE);
      keep_rendering = false;
    }
    if (!thread_pool)
      throw ::exception::InvalidThreadpoolStateException("Thread pool reference is invalid (nullptr).");
    return keep_rendering;
  }

  std::vector<std::future<void>> draw(HdrBufferStruct *buffers,
                                      const unsigned width_resolution,
                                      const unsigned height_resolution,
                                      NovaRenderEngineInterface *engine_instance,
                                      threading::ThreadPool *thread_pool,
                                      nova::nova_eng_internals &nova_internals) {

    AX_ASSERT(nova_resources != nullptr, "Scene descriptor is not initialized.");
    try {
      if (!validate(buffers, width_resolution, height_resolution, engine_instance, thread_pool, nova_internals))
        return {};
    } catch (const ::exception::GenericException &e) {
      throw;
    }

    std::vector<std::future<void>> futs;
    std::vector<Tile> tiles = divideByTiles(width_resolution,
                                            height_resolution,
                                            nova_internals.resource_manager->getEngineData().getTilesWidth(),
                                            nova_internals.resource_manager->getEngineData().getTilesHeight());
    for (auto &elem : tiles) {
      auto renderer_callback = [engine_instance](HdrBufferStruct *buffers, Tile &tile, nova::nova_eng_internals &nova_internals) {
        engine_instance->engine_render_tile(buffers, tile, nova_internals);
      };
      elem.sample_per_tile = nova_internals.resource_manager->getEngineData().getSampleIncrement();
      elem.image_total_height = height_resolution;
      elem.image_total_width = width_resolution;
      const std::string &tag = nova_internals.resource_manager->getEngineData().getTag();
      futs.push_back(thread_pool->addTask(tag, renderer_callback, buffers, elem, nova_internals));
    }

    return futs;
  }

}  // namespace nova
