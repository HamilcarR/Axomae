#ifndef LOADER_H
#define LOADER_H

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "GenericException.h"
#include "Image.h"
#include "Mesh.h"
#include "ResourceDatabaseManager.h"
#include "SceneHierarchy.h"
#include "constants.h"

/**
 * @file Loader.h
 * Implements a Loader class that will read mesh data and textures from disk
 *
 */

namespace IO {

  /**
   * @brief 3D Loader class
   */
  class Loader : public controller::IProgressManager {
   public:
    /**
     * @brief Construct a new Loader object
     *
     */
    explicit Loader(controller::ProgressStatus *progress_status);

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
     * @brief Loads all meshes from the GLB file.
     * @param filename GLB file path
     * @return std::vector<Mesh*>
     */
    std::pair<std::vector<Mesh *>, SceneTree> loadObjects(const char *filename);

    /**
     * @brief Loads an HDR image .
     * @param store if true , stores the loaded data in the Image db .
     * @return Image array
     */
    image::ImageHolder<float> loadHdr(const char *path, bool store = true);

    /**
     * @brief Writes an hdr file on disk , adds extension automatically , so no appending ".hdr" is necessary to the path
     */
    void writeHdr(const char *path, const image::ImageHolder<float> &image, bool flip = false);

   private:
    ResourceDatabaseManager *resource_database;
  };

}  // namespace IO

#endif
