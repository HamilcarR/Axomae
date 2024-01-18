#ifndef LOADER_H
#define LOADER_H

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "EnvmapController.h"
#include "GenericException.h"
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
  namespace exception {
    class LoadImagePathException : public GenericException {
     public:
      explicit LoadImagePathException(const std::string &path) {
        GenericException::saveErrorString(std::string("Failed loading the image : ") + path);
      }
    };

    class LoadImageDimException : public GenericException {
     public:
      explicit LoadImageDimException(int width, int height) {
        std::string dim = std::string("width : ") + std::to_string(width) + std::string(" height:") + std::to_string(height);
        GenericException::saveErrorString(std::string("Image dimensions error: ") + dim);
      }
    };

    class LoadImageChannelException : public GenericException {
     public:
      explicit LoadImageChannelException(int channels) {
        std::string chan = std::string("channels number : ") + std::to_string(channels);
        GenericException::saveErrorString(std::string("Image channel error: ") + chan);
      }
    };
  }  // namespace exception

  /**
   * @brief 3D Loader class
   */
  class Loader : public controller::IProgressManager {
   public:
    /**
     * @brief Construct a new Loader object
     *
     */
    explicit Loader(controller::ProgressStatus *prgoress_status);

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
     * @brief Loads an HDR image , and store it into it's database
     */
    void loadHdr(const char *path);

   private:
    ResourceDatabaseManager *resource_database;
  };

}  // namespace IO

#endif
