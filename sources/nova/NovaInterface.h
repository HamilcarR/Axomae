#ifndef NOVAINTERFACE_H
#define NOVAINTERFACE_H
#include "Ray.h"
#include "nova_camera.h"
#include "nova_texturing.h"

namespace nova {
  struct NovaResources {
    texturing::EnvmapResourcesHolder envmap_data;
    camera::CameraResourcesHolder camera_data{};
    int render_samples{};
  };

  class NovaRenderEngineLR;
  class NovaRenderEngineGR;
}  // namespace nova

template<class RENDERER_TYPE>
class NovaRenderEngineInterface {
 public:
  glm::vec4 sample_color(const nova::Ray &ray, const nova::NovaResources *nova_resources) {
    return (static_cast<RENDERER_TYPE *>(this))->engine_sample_color(ray, nova_resources);
  }
  void render_tile(float *dest_buffer,
                   int width_limit_low,
                   int width_limit_high,
                   int height_limit_low,
                   int height_limit_high,
                   const nova::NovaResources *nova_resources) {
    (static_cast<RENDERER_TYPE *>(this))
        ->engine_render_tile(dest_buffer, width_limit_low, width_limit_high, height_limit_low, height_limit_high, nova_resources);
  }
};

using NovaLRengineInterface = NovaRenderEngineInterface<nova::NovaRenderEngineLR>;
using NovaGRengineInterface = NovaRenderEngineInterface<nova::NovaRenderEngineGR>;
#endif  // NOVAINTERFACE_H
