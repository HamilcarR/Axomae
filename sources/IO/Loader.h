#ifndef LOADER_H
#define LOADER_H

#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "Mesh.h"
#include "ResourceDatabaseManager.h"
#include "SceneHierarchy.h"
#include "constants.h"

/**
 * @file Loader.h
 * Implements a Loader class that will read mesh data and textures from disk
 *
 */

namespace axomae {

  /**
   * @brief 3D Loader class
   */
  class Loader {
   public:
    /**
     * @brief Construct a new Loader object
     *
     */
    Loader();

    /**
     * @brief Destroy the Loader object
     *
     */
    ~Loader();

    /**
     * @brief Load a .glb file
     *
     * @param file Path of the 3D glb model
     * @return std::vector<Mesh*>
     */
    std::pair<std::vector<Mesh *>, SceneTree> load(const char *file);

    /**
     * @brief Loads a shader file into an std::string
     *
     * @param filename Path of the shader
     * @return std::string
     */
    std::string loadShader(const char *filename);

    /**
     * @brief Delete instance
     *
     */
    void close();

    /**
     * @brief Build the shaders and put them into the shader database
     *
     */
    void loadShaderDatabase();

    /**
     * @brief Build an environment map Mesh.
     *
     * @param is_glb_model Used to load a glb mesh as cubemap
     * @return Mesh*
     */
    Mesh *generateCubeMap(bool is_glb_model);

    /**
     * @brief Loads an environment map from the disk
     *
     * @return CubeMesh*
     */
    EnvironmentMap2DTexture *loadHdrEnvmap();

    /**
     * @brief Loads all meshes from the GLB file.
     * @param filename GLB file path
     * @return std::vector<Mesh*>
     */
    std::pair<std::vector<Mesh *>, SceneTree> loadObjects(const char *filename);

   protected:
    ResourceDatabaseManager *resource_database;
  };

}  // namespace axomae

#endif
