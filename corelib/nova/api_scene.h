#ifndef API_SCENE_H
#define API_SCENE_H
#include "api_common.h"

namespace nova {
  /**
   * @brief Opaque handle to the scene management class.
   * Stores internally the scene elements added.
   */
  class Scene {
   public:
    virtual ~Scene() = default;
    virtual ERROR_STATE addMesh(TrimeshPtr mesh, MaterialPtr material) = 0;
    /**
     * @brief Adds an HDR environment map to the internal collection of textures.
     * @return ID of the added texture.
     */
    virtual unsigned addEnvmap(TexturePtr envmap_texture) = 0;

    virtual ERROR_STATE useEnvmap(unsigned id) = 0;

    virtual Texture *getEnvmap(unsigned id) = 0;

    virtual const Texture *getEnvmap(unsigned id) const = 0;

    /**
     * @brief Get the Current Envmap Id
     *
     * @return int Envmap ID . Negative if not set.
     */
    virtual int getCurrentEnvmapId() const = 0;

    virtual ERROR_STATE setRootTransform(TransformPtr transform) = 0;
    /**
     * @brief Adds a camera to the scene.
     * @return ID of the camera
     */
    virtual unsigned addCamera(CameraPtr camera) = 0;

    /**
     * @brief Set the current camera ID.
     *
     * @param id ID of the camera used in the renderer.
     * @return ERROR_STATE
     *      - SUCCESS: If correct method sets up the ID correctly.
     *      - INVALID_ARGUMENT: If the ID is greater than the internal size of the camera pool.
     */
    virtual ERROR_STATE useCamera(unsigned id) = 0;

    /**
     * @brief Get the Current Camera's ID
     *
     * @return int Currently used Camera's ID. Negative if not set.
     */
    virtual int getCurrentCameraId() const = 0;

    virtual Camera *getCamera(unsigned id) = 0;

    virtual const Camera *getCamera(unsigned id) const = 0;

    /**
     * @brief Returns the total number of primitives in the scene, of a specific type.
     * @param type The type of primitives to count (e.g., TRIANGLE, SPHERE, BOX, NURB).
     * @returns Total number of primitives in the scene of the specified type.
     * @note Only triangle meshes are currently supported; other types may be added in the future.
     */
    virtual unsigned getPrimitivesNum(mesh::TYPE type) const = 0;

    /**
     * @brief Returns the total number of meshes in the scene, of a specific type.
     * @param type The type of meshes to count (e.g., TRIANGLE, SPHERE, BOX, NURB).
     * @returns Total number of meshes in the scene of the specified type.
     * @note Currently, only triangle meshes are supported. Future versions may support other mesh types.
     */
    virtual unsigned getMeshesNum(mesh::TYPE type) const = 0;

    /**
     * @brief Returns a const span of all triangle mesh pointers currently stored in the scene.
     * This collection contains all TriMeshPtrs that have been added to the scene via addMesh().
     * @returns A const span of all triangle mesh objects in the scene.
     */
    virtual CsteTriMeshCollection getTriangleMeshCollection() const = 0;

    /**
     * @brief Returns a const span of all material pointers currently stored in the scene.
     * This collection contains all MaterialPtrs that have been assigned to meshes via addMesh().
     * @param type The type of material to filter (e.g., TRIANGLE, SPHERE, BOX, NURB).
     * @returns A const span of all material objects in the scene that match the specified type.
     * @note Currently, materials are primarily assigned to triangle meshes. Other mesh types may support different material types in the future.
     */
    virtual CsteMaterialCollection getMaterialCollection(mesh::TYPE type) const = 0;

    /**
     * @brief Returns a const span of all camera pointers currently stored in the scene.
     * This collection contains all CameraPtrs that have been added via addCamera().
     * @returns A const span of all camera objects in the scene.
     */
    virtual CsteCameraCollection getCameraCollection() const = 0;

    virtual CsteTextureCollection getEnvmapCollection() const = 0;
  };

  ScenePtr create_scene();
}  // namespace nova
#endif
