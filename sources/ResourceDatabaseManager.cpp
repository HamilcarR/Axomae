
#include "../includes/ResourceDatabaseManager.h"

ResourceDatabaseManager *ResourceDatabaseManager::instance = nullptr;

ResourceDatabaseManager *ResourceDatabaseManager::getInstance() {
  if (instance == nullptr)
    instance = new ResourceDatabaseManager();
  return instance;
}

void ResourceDatabaseManager::destroyInstance() {
  if (instance)
    delete instance;
}

void ResourceDatabaseManager::purge() {
  if (texture_database)
    texture_database->purge();
  if (shader_database)
    shader_database->purge();
}

ResourceDatabaseManager::ResourceDatabaseManager() {
  texture_database = new TextureDatabase();
  shader_database = new ShaderDatabase();
}

ResourceDatabaseManager::~ResourceDatabaseManager() {
  if (texture_database)
    delete texture_database;
  if (shader_database)
    delete shader_database;
  shader_database = nullptr;
  texture_database = nullptr;
}

void ResourceDatabaseManager::cleanShaderDatabase() const {
  shader_database->clean();
}

void ResourceDatabaseManager::cleanTextureDatabase() const {
  texture_database->clean();
}

void ResourceDatabaseManager::purgeShaderDatabase() const {
  shader_database->purge();
}

void ResourceDatabaseManager::purgeTextureDatabase() const {
  texture_database->purge();
}