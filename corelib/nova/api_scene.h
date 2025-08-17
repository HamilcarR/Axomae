#ifndef API_SCENE_H
#define API_SCENE_H
#include "api_common.h"

namespace nova {
  class TriMesh;
  class Material;
  class Texture;
  class Camera;

  /**
   * @brief Opaque handle to the scene management class.
   * Stores internally the scene elements added.
   */
  class Scene {
   public:
    virtual ~Scene() = default;
    virtual ERROR_STATE addMesh(const TriMesh &mesh, const Material &material) = 0;
    virtual ERROR_STATE addEnvmap(const Texture &envmap_texture) = 0;
    virtual ERROR_STATE addCamera(const Camera &camera) = 0;
    virtual ERROR_STATE addRootTransform(const float transform[16]) = 0;
  };

  std::unique_ptr<Scene> create_scene();
}  // namespace nova
#endif
