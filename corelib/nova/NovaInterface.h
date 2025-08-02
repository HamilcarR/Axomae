#ifndef NOVAINTERFACE_H
#define NOVAINTERFACE_H
#include "manager/ManagerInternalStructs.h"

namespace nova {
  struct Tile;
  class NovaResourceManager;
  template<class T>
  struct RenderBuffers;
}  // namespace nova

class NovaRenderEngineInterface {
 public:
  virtual ~NovaRenderEngineInterface() = default;
  virtual void engine_render_tile(nova::RenderBuffers<float> *out_buffers, nova::Tile &tile, nova::nova_eng_internals &internal_structs) = 0;
};

#endif  // NOVAINTERFACE_H
