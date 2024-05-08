#ifndef NOVAINTERFACE_H
#define NOVAINTERFACE_H
#include "Ray.h"
#include "camera/nova_camera.h"
#include "rendering/nova_engine.h"
#include "scene/nova_scene.h"
#include "texturing/nova_texturing.h"

#include <map>
namespace nova {

  struct NovaResources {
    texturing::EnvmapResourcesHolder envmap_data{};
    camera::CameraResourcesHolder camera_data{};
    engine::EngineResourcesHolder renderer_data{};
    scene::SceneResourcesHolder scene_data{};
  };

  class NovaRenderEngineLR;
  class NovaRenderEngineGR;
  struct Tile;
}  // namespace nova

template<class RENDERER_TYPE>
class NovaRenderEngineInterface {
 public:
  glm::vec4 sample_color(const nova::Ray &ray, const nova::NovaResources *nova_resources) {
    return (static_cast<RENDERER_TYPE *>(this))->engine_sample_color(ray, nova_resources);
  }
  void render_tile(nova::HdrBufferStruct *out_buffers, const nova::Tile &tile, const nova::NovaResources *nova_resources) {
    (static_cast<RENDERER_TYPE *>(this))->engine_render_tile(out_buffers, tile, nova_resources);
  }
};

/* Use for local rays. */
using NovaLRengineInterface = NovaRenderEngineInterface<nova::NovaRenderEngineLR>;

/* Use for generalized rays.*/
using NovaGRengineInterface = NovaRenderEngineInterface<nova::NovaRenderEngineGR>;
#endif  // NOVAINTERFACE_H
