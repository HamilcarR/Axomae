#include "DrawEngine.h"
#include "integrator/Integrator.h"
#include "manager/NovaResourceManager.h"

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
  void NovaRenderEngineLR::engine_render_tile(HdrBufferStruct *buffers, Tile &tile, const NovaResourceManager *nova_resources) {
    AX_ASSERT(nova_resources, "Scene description is invalid.");
    AX_ASSERT(buffers, "Buffers not initialized.");

    integrator::integrator_dispatch(buffers, tile, nova_resources);
  }
}  // namespace nova
