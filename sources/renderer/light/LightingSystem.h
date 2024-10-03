#ifndef LIGHTINGSYSTEM_H
#define LIGHTINGSYSTEM_H

#include "LightInterface.h"
#include "Node.h"
#include "Shader.h"

/**
 * @file LightingSystem.h
 * Defines polymorphic lighting system.
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
  SceneTreeNode *parent;
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
  [[nodiscard]] bool checkUpdateFlag(LIGHTDATA_UPDATE_FLAGS flag) const { return update_flags & flag; }

  void asPbrColor(uint8_t red, uint8_t green, uint8_t blue) {
    diffuse_col = glm::vec3(red, green, blue);
    specular_col = glm::vec3(red, green, blue);
    ambiant_col = glm::vec3(red, green, blue);
  }

  void loadAttenuation(float cste, float lin, float quad) { attenuation = glm::vec3(cste, lin, quad); }
};

class AbstractLight : public SceneTreeNode, public LightInterface {
 public:
  enum TYPE : signed { DIRECTIONAL = 0, POINT = 1, SPOT = 2, AMBIANT = 3, HEMISPHERE = 4, QUAD = 5, AREA_TEXTURE = 6 };

 protected:
  TYPE type{};
  unsigned int id;
  glm::vec3 position{};
  glm::vec3 viewspace_position{};
  glm::vec3 specularColor{};
  glm::vec3 ambientColor{};
  glm::vec3 diffuseColor{};  // In case the renderer is PBR , we use only this variable for irradiance
  std::string light_struct_name{};
  float intensity{};

 protected:
  explicit AbstractLight(SceneTreeNode *parent = nullptr);

 public:
  AbstractLight(const AbstractLight &copy) = default;
  AbstractLight(AbstractLight &&move) noexcept = default;
  AbstractLight &operator=(const AbstractLight &copy) = default;
  AbstractLight &operator=(AbstractLight &&move) noexcept = default;
  ~AbstractLight() override = default;
  void setPosition(const glm::vec3 &pos) override { position = pos; }
  void setSpecularColor(const glm::vec3 &col) override { specularColor = col; }
  void setAmbiantColor(const glm::vec3 &col) override { ambientColor = col; }
  void setDiffuseColor(const glm::vec3 &col) override { diffuseColor = col; }
  [[nodiscard]] const glm::vec3 &getPosition() const override { return position; }
  [[nodiscard]] const glm::vec3 &getDiffuseColor() const override { return diffuseColor; }
  [[nodiscard]] const glm::vec3 &getAmbiantColor() const override { return ambientColor; }
  [[nodiscard]] const glm::vec3 &getSpecularColor() const override { return specularColor; }
  void setIntensity(float s) override { intensity = s; }
  [[nodiscard]] float getIntensity() const override { return intensity; }
  virtual void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index);
  virtual TYPE getType() { return type; }
  [[nodiscard]] unsigned int getID() const override { return id; }
  virtual void updateLightData(const LightData &data);
  void setID(unsigned light_id) override { id = light_id; }
};
/*****************************************************************************************************************/

/**
 * @class DirectionalLight
 */
class DirectionalLight : public AbstractLight {
 public:
  ~DirectionalLight() override = default;
  DirectionalLight(const DirectionalLight &copy) = default;
  DirectionalLight(DirectionalLight &&move) noexcept = default;
  DirectionalLight &operator=(const DirectionalLight &copy) = default;
  DirectionalLight &operator=(DirectionalLight &&move) noexcept = default;
  void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index) override;
  void updateLightData(const LightData &data) override;

 protected:
  explicit DirectionalLight(SceneTreeNode *parent = nullptr);
  explicit DirectionalLight(const LightData &light_data);
  DirectionalLight(glm::vec3 position, glm::vec3 color, float intensity, SceneTreeNode *parent = nullptr);
  DirectionalLight(
      glm::vec3 position, glm::vec3 ambientColor, glm::vec3 diffuseColor, glm::vec3 specularColor, float intensity, SceneTreeNode *parent = nullptr);
};

/*****************************************************************************************************************/

/**
 * @class PointLight
 */
class PointLight : public AbstractLight {
 protected:
  glm::vec3 attenuation{};

 public:
  ~PointLight() override = default;
  PointLight(const PointLight &copy) = default;
  PointLight(PointLight &&move) noexcept = default;
  PointLight &operator=(PointLight &&move) noexcept = default;
  PointLight &operator=(const PointLight &copy) = default;
  void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index) override;
  [[nodiscard]] const glm::vec3 &getAttenuation() const { return attenuation; }
  void setAttenuation(const glm::vec3 &atten) { attenuation = atten; }
  void updateLightData(const LightData &data) override;

 protected:
  explicit PointLight(SceneTreeNode *parent = nullptr);
  explicit PointLight(const LightData &data);
  PointLight(glm::vec3 position, glm::vec3 color, glm::vec3 attenuation_components, float intensity, SceneTreeNode *parent = nullptr);
  PointLight(glm::vec3 position,
             glm::vec3 ambientColor,
             glm::vec3 diffuseColor,
             glm::vec3 specularColor,
             glm::vec3 attenuation_compnents,
             float intensity,
             SceneTreeNode *parent = nullptr);
};

/*****************************************************************************************************************/

class SpotLight : public AbstractLight {

 protected:
  glm::vec3 direction{};
  glm::vec3 viewspace_direction{};
  float theta;

 public:
  ~SpotLight() override = default;
  SpotLight(const SpotLight &copy) = default;
  SpotLight(SpotLight &&move) noexcept = default;
  SpotLight &operator=(SpotLight &&move) noexcept = default;
  SpotLight &operator=(const SpotLight &copy) = default;
  void updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index) override;
  void updateLightData(const LightData &data) override;
  virtual void setDirection(glm::vec3 dir) { direction = dir; }
  virtual void setAngle(float angle) { theta = angle; }

 protected:
  explicit SpotLight(SceneTreeNode *parent = nullptr);
  explicit SpotLight(const LightData &data);
  SpotLight(glm::vec3 position, glm::vec3 direction, glm::vec3 color, float cutoff_angle, float intensity, SceneTreeNode *parent = nullptr);
  SpotLight(glm::vec3 position,
            glm::vec3 direction,
            glm::vec3 ambient,
            glm::vec3 diffuse,
            glm::vec3 specular,
            float cutoff_angle,
            float intensity,
            SceneTreeNode *parent = nullptr);
};

// TODO: [AX-33] Add Area lighting

#endif