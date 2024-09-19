#ifndef NOVAINTERFACE_H
#define NOVAINTERFACE_H
#include "engine/nova_engine.h"
#include "manager/ManagerInternalStructs.h"
namespace nova {
  struct Tile;
  class NovaResourceManager;
}  // namespace nova

class NovaRenderEngineInterface {
 public:
  virtual ~NovaRenderEngineInterface() = default;
  virtual void engine_render_tile(nova::HdrBufferStruct *out_buffers, nova::Tile &tile, nova::nova_eng_internals &internal_structs) = 0;
};

#endif  // NOVAINTERFACE_H
