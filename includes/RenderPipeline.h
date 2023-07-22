#ifndef RENDERPIPELINE_H
#define RENDERPIPELINE_H
#include "Renderer.h"
#include "Scene.h"
#include "ResourceDatabaseManager.h"

/**
 * @file RenderPipeline.h
 * This file implements the rendering steps for each techniques , like deferred rendering , depth peeling , texture baking  , occlusion culling etc.
 */
//TODO: [AX-8] Implements rendering pipeline




class Renderer;

/**
 * @class RenderPipeline 
 */
class RenderPipeline{
public:
    
    /**
    * @brief Create a new RenderPipeline object
    * @param renderer Pointer on the renderer object. 
    * @param resource_database Pointer on the resource database system.  
    */
    RenderPipeline(Renderer* renderer = nullptr , ResourceDatabaseManager* resource_database = nullptr); 

    /**
     * @brief Destroy the Render Pipeline object
     * 
     */
    virtual ~RenderPipeline(); 
    
    /**
     * @brief This method will bake an Environment map into a cubemap. 
     * A new cubemap texture will be created , and stored in the texture database. 
     * In addition , the texture will be assigned to the cubemap mesh. 
     * 
     * @param hdri_map  EnvironmentMapTexture that will be baked into a cubemap
     * @param cubemap_mesh The main scene's cubemap mesh .
     * @param width Width of the texture baked  
     * @param height Height of the texture baked
     */
    CubeMapMesh* bakeEnvmapToCubemap(EnvironmentMapTexture* hdri_map , unsigned width , unsigned height , GLViewer* gl_widget); 

    virtual void clean(); 
protected:
    Renderer* renderer ; 
    ResourceDatabaseManager* resource_database; 



};

















#endif