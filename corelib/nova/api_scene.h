#ifndef API_SCENE_H
#define API_SCENE_H
#include "api_common.h"

namespace nova {
  class TriMesh;
  class Material;
  class Texture;
  class Camera;
  class Transform;
  /**
   * @brief Opaque handle to the scene management class.
   * Stores internally the scene elements added.
   */
  class Scene {
   public:
    virtual ~Scene() = default;
    virtual ERROR_STATE addMesh(const TriMesh &mesh, const Material &material) = 0;
    /**
     * @brief Adds an HDR environment map to the internal collection of textures.
     * @return ID of the added texture.
     */
    virtual unsigned addEnvmap(const Texture &envmap_texture) = 0;
    virtual ERROR_STATE setRootTransform(const Transform &transform) = 0;
    /**
     * @brief
     * @return ID of camera
     */
    virtual unsigned addCamera(const Camera &camera) = 0;
  };

  std::unique_ptr<Scene> create_scene();
}  // namespace nova
#endif
