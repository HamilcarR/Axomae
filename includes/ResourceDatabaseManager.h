#ifndef RESOURCEDATABASEMANAGER_H
#define RESOURCEDATABASEMANAGER_H

#include "TextureDatabase.h"
#include "ShaderDatabase.h"
#include "RenderingDatabaseInterface.h"

/**
 * @file ResourceDatabaseManager.h
 * This file implements a singleton containing resources databases, like textures and shaders databases 
 * 
 */


/**
 * @brief This class contains all databases relative to resources used by the renderer. 
 * Every resource loaded using IO is managed from this class.
 * 
 */
class ResourceDatabaseManager{
public:
    static ResourceDatabaseManager* getInstance() ;

    /**
     * @brief This method deletes the stored databases objects . 
     * This method DOES NOT free the memory of what these databases contain. For this , use 
     * ResourceDatabaseManager::purge().
     * @see ResourceDatabaseManager::purge()
     */
    void destroyInstance();

    /**
     * @brief This method purges everything, deleting every resource stored , and the resources space taken GPU side
     * Additionally , will delete the singleton instance pointer , as well as the databases pointers.
     */
    void purge();

    /**
     * @brief  
     * 
     */
    void cleanTextureDatabase() const; 
    
    /**
     * @brief 
     * 
     */
    void cleanShaderDatabase() const; 
    
    /**
     * @brief 
     * 
     */
    void purgeTextureDatabase() const; 
    
    /**
     * @brief 
     * 
     */
    void purgeShaderDatabase() const; 
    /**
     * @brief Get the Texture Database object
     * 
     * @return TextureDatabase* 
     */
    TextureDatabase* getTextureDatabase() const {return texture_database;}

    /**
     * @brief Get the Shader Database object
     * 
     * @return ShaderDatabase* 
     */
    ShaderDatabase* getShaderDatabase() const {return shader_database;} 

private:
    ResourceDatabaseManager();  
    virtual ~ResourceDatabaseManager();
    ResourceDatabaseManager(const ResourceDatabaseManager&) = delete ; 
    ResourceDatabaseManager operator=(const ResourceDatabaseManager&) = delete; 

private: 
    static ResourceDatabaseManager *instance;
    TextureDatabase *texture_database; 
    ShaderDatabase  *shader_database; 
};











#endif