#ifndef API_SCENE_H
#define API_SCENE_H
#include "api_common.h"

namespace nova {
  class NvAbstractTriMesh;
  class NvAbstractMaterial;
  class NvAbstractTexture;
  class Camera;

  /**
   * @brief Opaque handle to the scene management class.
   * Stores internally the scene elements added.
   */
  class NvAbstractScene {
   public:
    virtual ~NvAbstractScene() = default;
    virtual ERROR_STATE addMesh(const NvAbstractTriMesh &mesh, const NvAbstractMaterial &material) = 0;
    virtual ERROR_STATE addEnvmap(const NvAbstractTexture &envmap_texture) = 0;
  };

  std::unique_ptr<NvAbstractScene> create_scene();
}  // namespace nova
#endif
