#ifndef BAKERENDERDATA_H
#define BAKERENDERDATA_H
#include "Image.h"
#include "nova/bake.h"

#include <manager/NovaResourceManager.h>

/**
 *@brief structures related to the management of offline renderers resources
 */
namespace controller {
  struct bake_temp_buffers {
    image::ThumbnailImageHolder<float> image_holder;
    std::vector<float> accumulator;
    std::vector<float> partial;
    std::vector<float> depth;
  };

  struct NovaBakingStructure {
    bake_temp_buffers bake_buffers;
    std::unique_ptr<QWidget> spawned_window;
    nova_baker_utils::render_scene_data nova_render_scene;
    std::thread rendering_thread;

    void reinitialize() {
      auto &nova_resource_manager = nova_render_scene.nova_resource_manager;
      if (nova_resource_manager)
        nova_resource_manager->getEngineData().is_rendering = false;
      if (rendering_thread.joinable())
        rendering_thread.join();
      /* Clear the render buffers. */
      bake_buffers.image_holder.clear();
      bake_buffers.accumulator.clear();
      bake_buffers.partial.clear();
      /* Remove reference to the widget. */
      spawned_window = nullptr;
    }
  };

}  // namespace controller

#endif  // BAKERENDERDATA_H
