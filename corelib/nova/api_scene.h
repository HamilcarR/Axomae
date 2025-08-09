#ifndef API_SCENE_H
#define API_SCENE_H
#include "api_common.h"
namespace nova {
  class NvAbstractTriMesh;
  class NvAbstractMaterial;
  class NvAbstractScene {
   public:
    virtual ~NvAbstractScene() = default;
    virtual ERROR_STATE addMesh(const NvAbstractTriMesh &mesh, const NvAbstractMaterial &material);
  };

  inline std::unique_ptr<NvAbstractScene> create_scene();
}  // namespace nova
#endif
