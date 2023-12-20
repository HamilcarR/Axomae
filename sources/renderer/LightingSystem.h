#ifndef LIGHTINGSYSTEM_H
#define LIGHTINGSYSTEM_H

#include "Node.h"
#include "Shader.h"
#include "utils_3D.h"

/**
 * @file LightingSystem.h
 * Defines polymorphic lighting system , as directional , spot lights and point lights
 */

class LightData {
 public:
  enum LIGHTDATA_UPDATE_FLAGS : unsigned {
    POSITION_UPDATE = 1,
    DIRECTION_UPDATE = 1 << 2,
    ATTENUATION_UPDATE = 1 << 3,
    DIFFUSE_UPDATE = 1 << 4,
    SPECULAR_UPDATE = 1 << 5,
    AMBIANT_UPDATE = 1 << 6,
    INTENSITY_UPDATE = 1 << 7,
    THETA_UPDATE = 1 << 8
  };
  ISceneNode *parent;
  glm::vec3 position;
  glm::vec3 direction;
  glm::vec3 attenuation;
  glm::vec3 diffuse_col;
  glm::vec3 specular_col;
  glm::vec3 ambiant_col;
  float intensity;
  float theta;
  std::string name;
  uint16_t update_flags = 0;

  /*Helper functions */
  void enableUpdateFlag(LIGHTDATA_UPDATE_FLAGS flag) { update_flags |= flag; }

  void disableUpdateFlag(LIGHTDATA_UPDATE_FLAGS flag) { update_flags &= ~flag; }

  bool checkUpdateFlag(LIGHTDATA_UPDATE_FLAGS flag) const { return update_flags & flag; }

  void asPbrColor(uint8_t red, uint8_t green, uint8_t blue) {
    diffuse_col = glm::vec3(red, green, blue);
    specular_col = glm::vec3(red, green, blue);
    ambiant_col = glm::vec3(red, green, blue);
  }

  void loadAttenuation(float cste, float lin, float quad) { attenuation = glm::vec3(cste, lin, quad); }
};

/**
 * @class AbstractLight
 * @brief Abstract class providing an interface for the base light system
 *
 */
class AbstractLight : public SceneTreeNode {
 public:
  /**
   * @brief Type of the light
   *
   */
  enum TYPE : signed {
    DIRECTIONAL = 0, /**<Directional light . Only a direction*/
    POINT = 1,       /**<Point light*/
    SPOT = 2,        /**<Spot light*/
    AMBIANT = 3,     /**<Ambiant lighting*/
    HEMISPHERE = 4,
    QUAD = 5,
    AREA_TEXTURE = 6
  };

  virtual ~AbstractLight() {}

  /**
   * @brief Set the Position of the ligh
   *
   * @param pos glm::vec3 position
   */
  virtual void setPosition(glm::vec3 pos) { position = pos; }

  /**
   * @brief Set the Specular Color of the light
   *
   * @param col glm::vec3 color
   */
  virtual void setSpecularColor(glm::vec3 col) { specularColor = col; }

  /**
   * @brief Set the Ambiant Color
   *
   * @param col
   */
  virtual void setAmbiantColor(glm::vec3 col) { ambientColor = col; }

  /**
   * @brief Set the Diffuse Color
   *
   * @param col
   */
  virtual void setDiffuseColor(glm::vec3 col) { diffuseColor = col; }
  /**
   * @brief Get the light's position
   *
   * @return glm::vec3
   */
  virtual glm::vec3 getPosition() { return position; }

  /**
   * @brief Get the Diffuse Color
   *
   * @return glm::vec3
   */
  virtual glm::vec3 getDiffuseColor() { return diffuseColor; }

  /**
   * @brief Get the Ambiant Color
   *
   * @return glm::vec3
   */
  virtual glm::vec3 getAmbiantColor() { return ambientColor; }

  /**
   * @brief Get the Specular Color
   *
   * @return glm::vec3
   */
  virtual glm::vec3 getSpecularColor() { return specularColor; }

  /**
   * @brief Set the Intensity value
   *
   * @param s New intensity
   */
  virtual void setIntensity(float s) { intensity = s; }

  /**
   * @brief Get the Intensity value
   *
   */
  virtual float getIntensity() { return intensity; }

  /**
   * @brief Updates the uniforms values of lights in the shader
   *
   * @param shader Pointer to the shader
   * @param view view matrix for light transformations
   * @param index Index position in the array corresponding on the type of the light in the database
   */
  virtual void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index);

  /**
   * @brief Get the type of the light
   *
   * @return TYPE
   * @see AbstractClass::TYPE
   */
  virtual TYPE getType() { return type; }

  /**
   * @brief Returns the ID of the light
   *
   * @return unsigned int
   */
  virtual unsigned int getID() { return id; }

  /**
   * @brief Updates this light's internal data with new values
   *
   * @param data LightData structure .
   */
  virtual void updateLightData(const LightData &data);

  void setID(const unsigned light_id) { id = light_id; }

 protected:
  /**
   * @brief Construct a new Abstract Light object
   *
   * @param parent Predecessor in the scene graph
   */
  AbstractLight(ISceneNode *parent = nullptr);

 protected:
  TYPE type;
  unsigned int id;
  glm::vec3 position;
  glm::vec3 viewspace_position;
  glm::vec3 specularColor;
  glm::vec3 ambientColor;
  glm::vec3 diffuseColor;  // In case the renderer is PBR , we use only this variable for irradiance
  std::string light_struct_name;
  float intensity;
};
/*****************************************************************************************************************/

/**
 * @class DirectionalLight
 * @brief Class declaration of the Directional light
 */
class DirectionalLight : public AbstractLight {

 protected:
  /**
   * @brief Construct a new Directional Light object
   * @param parent Predecessor in the scene graph
   */
  DirectionalLight(ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Directional Light object
   *
   * @param position Position of the light
   * @param color General color of the light
   * @param intensity Intensity of the light
   * @param parent Predecessor in the scene graph
   */
  DirectionalLight(glm::vec3 position, glm::vec3 color, float intensity, ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Directional Light
   *
   * @param position Position of the light in world space
   * @param ambientColor Ambient color
   * @param diffuseColor Diffuse color
   * @param specularColor Specular color
   * @param intensity Intensity of the light
   * @param parent Predecessor in the scene graph
   */
  DirectionalLight(
      glm::vec3 position, glm::vec3 ambientColor, glm::vec3 diffuseColor, glm::vec3 specularColor, float intensity, ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Directional Light
   *
   * @param light_data Light data . Will only take into account data that are related to directional lights
   * @param parent Predecessor in the scene graph .
   */
  DirectionalLight(const LightData &light_data);

 public:
  /**
   * @brief Destroy the Directional Light object
   *
   */
  virtual ~DirectionalLight();

  /**
   * @brief Computes the directional light direction and stores it into the viewspace_position property , then calls
   * AbstractLight::updateShaderData() which updates uniforms accordingly
   * @overload void AbstractLight::updateShaderData(Shader* shader , glm::mat4& view , unsigned int index) const ;
   */
  virtual void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index);

  virtual void updateLightData(const LightData &data) override;

 protected:
};

/*****************************************************************************************************************/

/**
 * @class PointLight
 * @brief Point light declaration
 *
 */
class PointLight : public AbstractLight {

 protected:
  /**
   * @brief Construct a new Point Light object
   * @param parent Predecessor in the scene graph
   *
   */
  PointLight(ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Point Light object
   *
   * @param position Position of the light
   * @param color Color of the light
   * @param attenuation_components glm::vec3 representing the 3 attenuation components of a point light ,
   * attenuation_components.x being the constant component , attenuation_components.y the linear , and the last one is
   * the quadratic.
   * @param intensity Intensity of the light
   * @param parent Predecessor in the scene graph
   */
  PointLight(glm::vec3 position, glm::vec3 color, glm::vec3 attenuation_components, float intensity, ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Point Light object
   *
   * @param position Position of the light
   * @param ambientColor The ambient color
   * @param diffuseColor The diffuse color
   * @param specularColor The specular color
   * @param attenuation_components glm::vec3 representing the 3 attenuation components of a point light ,
   * attenuation_components.x being the constant component , attenuation_components.y the linear , and the last one is
   * the quadratic.
   * @param intensity Intensity of the light
   * @param parent Predecessor in the scene graph
   */
  PointLight(glm::vec3 position,
             glm::vec3 ambientColor,
             glm::vec3 diffuseColor,
             glm::vec3 specularColor,
             glm::vec3 attenuation_compnents,
             float intensity,
             ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Point Light object
   *
   * @param data
   * @param parent
   */
  PointLight(const LightData &data);

 public:
  /**
   * @brief Destroy the Point Light object
   *
   */
  virtual ~PointLight();

  /**
   * @brief Enable point light values in shader's uniforms
   *
   * @overload void AbstractLight::updateShaderData(Shader* shader , glm::mat4& view , unsigned int index) const ;
   */
  virtual void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index) override;

  glm::vec3 getAttenuation() const { return attenuation; }

  void setAttenuation(glm::vec3 atten) { attenuation = atten; }

  virtual void updateLightData(const LightData &data) override;

 protected:
  glm::vec3 attenuation; /**<Constant, linear, and quadratic attenuation values*/
};

/*****************************************************************************************************************/

class SpotLight : public AbstractLight {

 protected:
  /**
   * @brief Construct a new Spot Light object
   * @param parent Predecessor in the scene graph
   *
   */
  SpotLight(ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Spot Light object
   *
   * @param position
   * @param direction
   * @param color
   * @param cutoff_angle
   * @param intensity
   * @param parent Predecessor in the scene graph
   */
  SpotLight(glm::vec3 position, glm::vec3 direction, glm::vec3 color, float cutoff_angle, float intensity, ISceneNode *parent = nullptr);

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
   * @param parent Predecessor in the scene graph
   */
  SpotLight(glm::vec3 position,
            glm::vec3 direction,
            glm::vec3 ambient,
            glm::vec3 diffuse,
            glm::vec3 specular,
            float cutoff_angle,
            float intensity,
            ISceneNode *parent = nullptr);

  /**
   * @brief Construct a new Spot Light object
   *
   * @param data
   * @param ancestor
   */
  SpotLight(const LightData &data);

 public:
  /**
   * @brief
   *
   * @overload void AbstractLight::updateShaderData(Shader* shader , glm::mat4& view , unsigned int index) const ;
   */
  virtual void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index);

  virtual void updateLightData(const LightData &data) override;

  virtual void setDirection(glm::vec3 dir) { direction = dir; }

  virtual void setAngle(float angle) { theta = angle; }

 protected:
  glm::vec3 direction;           /**<Direction of the light cone*/
  glm::vec3 viewspace_direction; /**<Direction in view space*/
  float theta;                   /**<Angle of the light cone , in degrees*/
};

// TODO: [AX-33] Add Area lighting

#endif