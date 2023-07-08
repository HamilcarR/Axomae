#ifndef RESOURCEDATABASEMANAGER_H
#define RESOURCEDATABASEMANAGER_H

#include "TextureDatabase.h"
#include "ShaderDatabase.h"
#include "RenderingDatabaseInterface.h"

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
     * @brief Proceeds with a soft clean on the texture database 
     * 
     */
    void cleanTextureDatabase() const; 
    
    /**
     * @brief Proceeds with a soft clean on the shader database 
     * 
     */
    void cleanShaderDatabase() const; 
    
    /**
     * @brief Purge the entire texture database 
     * 
     */
    void purgeTextureDatabase() const; 
    
    /**
     * @brief Purge the entire shader database 
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

    /**
     * @brief Construct a new Resource Database Manager object
     * 
     */
    ResourceDatabaseManager();  
    
    /**
     * @brief Destroy the Resource Database Manager object
     * 
     */
    virtual ~ResourceDatabaseManager();
    
    /**
     * @brief Construct a new Resource Database Manager object
     * 
     */
    ResourceDatabaseManager(const ResourceDatabaseManager&) = delete ; 
    
    /**
     * @brief 
     * 
     * @return ResourceDatabaseManager 
     */
    ResourceDatabaseManager operator=(const ResourceDatabaseManager&) = delete; 

private: 
    static ResourceDatabaseManager *instance;   /*<Instance of this ResourceDatabaseManager*/
    TextureDatabase *texture_database;          /*<Pointer on the texture database*/ 
    ShaderDatabase  *shader_database;           /*<Pointer on the shader database*/
};











#endif