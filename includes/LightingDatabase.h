#ifndef LIGHTINGDATABASE_H
#define LIGHTINGDATABASE_H

#include "utils_3D.h"
#include "LightingSystem.h"



/**
 * @file LightingDatabase.h
 * Database of lights used in the scene.  
 * 
 */



/**
 * @class LightingDatabase
 * @brief Class declaration of the light database system
 */
class LightingDatabase{
public:
    /**
     * @brief Construct a new Lighting Database object
     * 
     */
    LightingDatabase(); 
    
    /**
     * @brief Destroy the Lighting Database object
     * 
     */
    virtual ~LightingDatabase(); 

    /**
     * @brief Add a new light in the scene
     * 
     * @param light 
     */
    virtual bool addLight(AbstractLight* light);

    /**
     * @brief Get all lights of certain type
     * 
     * @param type Type of light we want
     * @return const std::vector<AbstractLight*>& Collection of light pointers
     */
    const std::vector<AbstractLight*>& getLightsArrayByType(AbstractLight::TYPE type) ;

    /**
     * @brief Deletes all pointers and the subsequent vector array of lights that matches the type
     * 
     * @param type Type to be deleted
     */
    void eraseLightsArray(AbstractLight::TYPE type); 

    /**
     * @brief Empty the whole database , and deletes all objects stored  
     * 
     */
    void clearDatabase();

    /**
     * @brief Update shader uniforms for all lights
     * 
     * @param shader Shader pointer to send data to 
     * @param view  View matrix for light transformations
     */
    virtual void updateShadersData(Shader* shader , glm::mat4& view); 
    /**
     * @brief Update shader uniforms only for lights of "type" 
     * 
     * @param type Type of lights to send to the shader 
     * @param shader Shader applied to the mesh 
     * @param view View matrix for light transformations
     */
    virtual void updateShadersData(AbstractLight::TYPE type , Shader* shader , glm::mat4& view);
protected:
    std::map<AbstractLight::TYPE , std::vector<AbstractLight*>> light_database ;    /**<Map of all lights in the scene. The map key stored is the type of light*/



};

#endif