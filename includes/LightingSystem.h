#ifndef LIGHTINGSYSTEM_H
#define LIGHTINGSYSTEM_H

#include "Shader.h"
#include "utils_3D.h"

/**
 * @file LightingSystem.h
 * Defines polymorphic lighting system , as directional , spot lights and point lights  
 */



/**
 * @class AbstractLight
 * @brief Abstract class providing an interface for the base light system 
 * 
 */
class AbstractLight{
public:

    /**
     * @brief Type of the light
     * 
     */
    enum TYPE:signed
    {
        DIRECTIONAL = 0 ,       /**<Directional light . Only a direction*/
        POINT = 1 ,             /**<Point light*/
        SPOT = 2 ,              /**<Spot light*/
        AMBIANT = 3,             /**<Ambiant lighting*/
        HEMISPHERE = 4 , 
        QUAD = 5 , 
        AREA_TEXTURE = 6
    };
     
    virtual ~AbstractLight(){}
   
    /**
     * @brief Set the Position of the ligh
     * 
     * @param pos glm::vec3 position
     */
    virtual void setPosition(glm::vec3 pos){position = pos ; }

    /**
     * @brief Set the Specular Color of the light
     * 
     * @param col glm::vec3 color 
     */
    virtual void setSpecularColor(glm::vec3 col){specularColor = col;}

    /**
     * @brief Set the Ambiant Color 
     * 
     * @param col 
     */
    virtual void setAmbiantColor(glm::vec3 col){ambientColor = col;}

    /**
     * @brief Set the Diffuse Color 
     * 
     * @param col 
     */
    virtual void setDiffuseColor(glm::vec3 col){diffuseColor = col;}
    /**
     * @brief Get the light's position
     * 
     * @return glm::vec3 
     */
    virtual glm::vec3 getPosition(){return position;}

    /**
     * @brief Get the Diffuse Color 
     * 
     * @return glm::vec3 
     */
    virtual glm::vec3 getDiffuseColor(){return diffuseColor;}

    /**
     * @brief Get the Ambiant Color 
     * 
     * @return glm::vec3 
     */
    virtual glm::vec3 getAmbiantColor(){return ambientColor;}

    /**
     * @brief Get the Specular Color
     * 
     * @return glm::vec3 
     */
    virtual glm::vec3 getSpecularColor(){return specularColor;}  
    
    /**
     * @brief Set the Intensity value
     * 
     * @param s New intensity
     */
    virtual void setIntensity(float s){intensity = s;}
    
    /**
     * @brief Get the Intensity value
     * 
     */
    virtual float getIntensity(){return intensity;}

    /**
     * @brief Updates the uniforms values of lights in the shader
     * 
     * @param shader Pointer to the shader
     * @param modelview Modelview matrix for light transformations
     * @param index Index position in the array corresponding on the type of the light in the database 
     */
    virtual void updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) const ; 

    /**
     * @brief Get the type of the light
     * 
     * @return TYPE 
     * @see AbstractClass::TYPE
     */
    virtual TYPE getType() const = 0 ;

    /**
     * @brief Returns the ID of the light
     * 
     * @return unsigned int 
     */
    virtual unsigned int getID() {return id ; } 

protected:
    AbstractLight(){}
protected:
    TYPE type; 
    unsigned int id ;  
    glm::vec3 position ; 
    glm::vec3 specularColor; 
    glm::vec3 ambientColor; 
    glm::vec3 diffuseColor; 
    float intensity ; 
};
/*****************************************************************************************************************/

/**
 * @class DirectionalLight
 * @brief Class declaration of the Directional light
 */
class DirectionalLight : public AbstractLight {
public:

    /**
     * @brief Construct a new Directional Light object
     * 
     */
    DirectionalLight() ;

    /**
     * @brief Construct a new Directional Light object
     * 
     * @param _position Position of the light
     * @param color General color of the light 
     * @see Shader
     */
    DirectionalLight(glm::vec3 _position , glm::vec3 color , float _intensity);

    /**
     * @brief Destroy the Directional Light object
     * 
     */
    virtual ~DirectionalLight();

    /**
     * @brief Get the Type of the light
     * 
     * @return TYPE 
     */
    virtual TYPE getType() const {return type;}

    

protected:


};
/*****************************************************************************************************************/





















#endif