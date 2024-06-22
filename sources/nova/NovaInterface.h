#ifndef NOVAINTERFACE_H
#define NOVAINTERFACE_H
#include "Ray.h"
#include "camera/nova_camera.h"
#include "rendering/nova_engine.h"
#include "scene/nova_scene.h"
#include "texturing/nova_texturing.h"

#include <map>
namespace nova {

  class NovaResources {
   public:
    texturing::TextureRawData envmap_data{};
    texturing::TextureResourcesHolder textures_data{};
    camera::CameraResourcesHolder camera_data{};
    engine::EngineResourcesHolder renderer_data{};
    scene::SceneResourcesHolder scene_data{};
    scene::Accelerator acceleration_structure{};
  };

  struct Tile;
}  // namespace nova

class NovaRenderEngineInterface {
 public:
  virtual ~NovaRenderEngineInterface() = default;
  virtual glm::vec4 engine_sample_color(const nova::Ray &ray, const nova::NovaResources *nova_resources, int depth) = 0;
  virtual void engine_render_tile(nova::HdrBufferStruct *out_buffers, nova::Tile &tile, const nova::NovaResources *nova_resources) = 0;
};

#endif  // NOVAINTERFACE_H
