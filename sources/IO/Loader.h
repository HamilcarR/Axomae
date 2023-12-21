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
     * @brief Load a .glb file
     *
     * @param file Path of the 3D glb model
     * @return std::vector<Mesh*>
     */
    std::pair<std::vector<Mesh *>, SceneTree> load(const char *file);

    /**
     * @brief Loads a text file , with new lines
     *
     * @param filename Path of the file
     * @return std::string
     */
    std::string loadTextFile(const char *filename);

    /**
     * @brief Delete instance
     *
     */
    void close();

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
