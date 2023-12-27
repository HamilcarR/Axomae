#ifndef RESOURCEDATABASEMANAGER_H
#define RESOURCEDATABASEMANAGER_H

#include "INodeDatabase.h"
#include "ImageDatabase.h"
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
class ResourceDatabaseManager {
 public:
  static ResourceDatabaseManager &getInstance();

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
  TextureDatabase &getTextureDatabase() const { return *texture_database; }
  ShaderDatabase &getShaderDatabase() const { return *shader_database; }
  INodeDatabase &getNodeDatabase() const { return *node_database; }
  HdrImageDatabase &getHdrDatabase() const { return *hdr_database; }

 private:
  /**
   * @brief Construct a new Resource Database Manager object
   *
   */
  ResourceDatabaseManager();

  /**
   * @brief Construct a new Resource Database Manager object
   *
   */
  ResourceDatabaseManager(const ResourceDatabaseManager &) = delete;

  /**
   * @brief
   *
   * @return ResourceDatabaseManager
   */
  ResourceDatabaseManager operator=(const ResourceDatabaseManager &) = delete;

 private:
  std::unique_ptr<TextureDatabase> texture_database; /*<Pointer on the texture database*/
  std::unique_ptr<ShaderDatabase> shader_database;   /*<Pointer on the shader database*/
  std::unique_ptr<INodeDatabase> node_database;      /*<Pointer on the node database*/
  std::unique_ptr<HdrImageDatabase> hdr_database;    /*<Raw HDR images database*/
};

#endif