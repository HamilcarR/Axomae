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
    virtual void updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) ; 

    /**
     * @brief Get the type of the light
     * 
     * @return TYPE 
     * @see AbstractClass::TYPE
     */
    virtual TYPE getType() {return type;}

    /**
     * @brief Returns the ID of the light
     * 
     * @return unsigned int 
     */
    virtual unsigned int getID() {return id ; } 

protected:
    TYPE type; 
    unsigned int id ;  
    glm::vec3 position ;
    glm::vec3 viewspace_position;  
    glm::vec3 specularColor; 
    glm::vec3 ambientColor; 
    glm::vec3 diffuseColor;
    std::string light_struct_name ;  
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
     * @param position Position of the light
     * @param color General color of the light 
     * @param intensity Intensity of the light
     */
    DirectionalLight(glm::vec3 position , glm::vec3 color , float intensity);

    /**
     * @brief Construct a new Directional Light 
     * 
     * @param position Position of the light in world space
     * @param ambientColor Ambient color
     * @param diffuseColor Diffuse color 
     * @param specularColor Specular color 
     * @param intensity Intensity of the light
     */
    DirectionalLight(glm::vec3 position , glm::vec3 ambientColor , glm::vec3 diffuseColor , glm::vec3 specularColor , float intensity);  
 
    /**
     * @brief Destroy the Directional Light object
     * 
     */
    virtual ~DirectionalLight();
 
    /**
     * @brief Computes the directional light direction and stores it into the viewspace_position property , then calls AbstractLight::updateShaderData() which updates uniforms accordingly
     * @overload void AbstractLight::updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) const ; 
     */
    virtual void updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) ; 

protected:
};

/*****************************************************************************************************************/

/**
 * @class PointLight
 * @brief Point light declaration
 * 
 */
class PointLight : public AbstractLight{
public:
    /**
     * @brief Construct a new Point Light object
     * 
     */
    PointLight(); 
    
    /**
     * @brief Construct a new Point Light object
     * 
     * @param position Position of the light
     * @param color Color of the light
     * @param attenuation_components glm::vec3 representing the 3 attenuation components of a point light , attenuation_components.x being the constant component , attenuation_components.y the linear , and the last one is the quadratic.
     * @param intensity Intensity of the light
     */
    PointLight(glm::vec3 position , glm::vec3 color , glm::vec3 attenuation_components , float intensity); 
    
    /**
     * @brief Construct a new Point Light object
     * 
     * @param position Position of the light 
     * @param ambientColor The ambient color
     * @param diffuseColor The diffuse color
     * @param specularColor The specular color
     * @param attenuation_components glm::vec3 representing the 3 attenuation components of a point light , attenuation_components.x being the constant component , attenuation_components.y the linear , and the last one is the quadratic.
     * @param intensity Intensity of the light 
     */
    PointLight(glm::vec3 position , glm::vec3 ambientColor , glm::vec3 diffuseColor , glm::vec3 specularColor , glm::vec3 attenuation_compnents , float intensity); 
    
    /**
     * @brief Destroy the Point Light object
     * 
     */
    virtual ~PointLight(); 

    /**
     * @brief Enable point light values in shader's uniforms
     *
     * @overload void AbstractLight::updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) const ;
     */
    virtual void updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) override ; 
protected:
    glm::vec3 attenuation ;         /**<Constant, linear, and quadratic attenuation values*/
};

/*****************************************************************************************************************/

class SpotLight : public AbstractLight{
public:
    
    /**
     * @brief Construct a new Spot Light object
     * 
     */
    SpotLight(); 
    
    /**
     * @brief Construct a new Spot Light object
     * 
     * @param position 
     * @param direction 
     * @param color 
     * @param cutoff_angle 
     * @param intensity 
     */
    SpotLight(glm::vec3 position , glm::vec3 direction , glm::vec3 color , float cutoff_angle , float intensity); 
    
    /**
     * @brief Construct a new Spot Light object
     * 
     * @param position 
     * @param direction 
     * @param ambient 
     * @param diffuse 
     * @param specular 
     * @param cutoff_angle 
     * @param intensity 
     */
    SpotLight( glm::vec3 position , glm::vec3 direction , glm::vec3 ambient , glm::vec3 diffuse , glm::vec3 specular , float cutoff_angle ,  float intensity);

    /**
     * @brief 
     * 
     * @overload void AbstractLight::updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) const ;  
     */
    virtual void updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) ; 

protected:
    glm::vec3 direction ;           /**<Direction of the light cone*/
    glm::vec3 viewspace_direction;  /**<Direction in modelview space*/
    float theta;                    /**<Angle of the light cone , in degrees*/

};



//TODO: Add Area lighting

#endif