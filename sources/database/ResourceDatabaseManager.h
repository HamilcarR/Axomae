#ifndef RESOURCEDATABASEMANAGER_H
#define RESOURCEDATABASEMANAGER_H

#include "INodeDatabase.h"
#include "ImageDatabase.h"
#include "OP_ProgressStatus.h"
#include "ShaderDatabase.h"
#include "TextureDatabase.h"
/**
 * @file ResourceDatabaseManager.h
 * @brief This file implements a singleton containing resources databases, like textures and shaders databases
 *
 */

/**
 * @brief This class contains all databases relative to resources used by the renderer.
 * Every resource loaded using IO is managed from this class.
 *
 */
class ResourceDatabaseManager final : public controller::IProgressManager {  // TODO : Not the best design choice , replace with proper class
 public:
  static ResourceDatabaseManager &getInstance();
  ResourceDatabaseManager(const ResourceDatabaseManager &) = delete;
  ResourceDatabaseManager &operator=(const ResourceDatabaseManager &) = delete;
  ResourceDatabaseManager &operator=(ResourceDatabaseManager &&) = delete;
  ResourceDatabaseManager(ResourceDatabaseManager &&) = delete;
  /**
   * @brief This method purges everything, deleting every resource stored , and the resources space taken GPU side
   * Additionally , will delete the singleton instance pointer , as well as the databases pointers.
   */
  void purge();
  void clean();
  void cleanTextureDatabase();
  void cleanHdrDatabase();
  void cleanShaderDatabase();
  void cleanNodeDatabase();
  void purgeTextureDatabase();
  void purgeHdrDatabase();
  void purgeShaderDatabase();
  void purgeNodeDatabase();
  void cleanRawImgDatabase();
  void purgeRawImgDatabase();
  [[nodiscard]] TextureDatabase &getTextureDatabase() const { return *texture_database; }
  [[nodiscard]] ShaderDatabase &getShaderDatabase() const { return *shader_database; }
  [[nodiscard]] INodeDatabase &getNodeDatabase() const { return *node_database; }
  [[nodiscard]] HdrImageDatabase &getHdrDatabase() const { return *hdr_database; }
  [[nodiscard]] RawImageDatabase &getRawImgdatabase() const { return *image_database; }

  void setProgressManager(controller::ProgressStatus *progress_manager) override;
  void initializeDatabases(controller::ProgressStatus *progress_manager = nullptr);

 private:
  /**
   * @brief Construct a new Resource Database Manager object
   *
   */
  ResourceDatabaseManager();
  ~ResourceDatabaseManager() = default;

 private:
  std::unique_ptr<TextureDatabase> texture_database; /*<Pointer on the texture database*/
  std::unique_ptr<ShaderDatabase> shader_database;   /*<Pointer on the shader database*/
  std::unique_ptr<INodeDatabase> node_database;      /*<Pointer on the node database*/
  std::unique_ptr<HdrImageDatabase> hdr_database;    /*<Raw HDR images database*/
  std::unique_ptr<RawImageDatabase> image_database;  /*<Raw texture images database*/
};

#endif