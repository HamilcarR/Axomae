#ifndef NOVAINTERFACE_H
#define NOVAINTERFACE_H
#include "camera/nova_camera.h"
#include "ray/Ray.h"
#include "rendering/nova_engine.h"
#include "scene/nova_scene.h"
#include "texturing/nova_texturing.h"

#include <map>
namespace nova {
  struct Tile;
  class NovaResourceManager;
}  // namespace nova

class NovaRenderEngineInterface {
 public:
  virtual ~NovaRenderEngineInterface() = default;
  virtual void engine_render_tile(nova::HdrBufferStruct *out_buffers, nova::Tile &tile, const nova::NovaResourceManager *nova_resources) = 0;
};

#endif  // NOVAINTERFACE_H
