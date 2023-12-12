#include "../includes/LightingSystem.h"
#include "../includes/UniformNames.h"

AbstractLight::AbstractLight(ISceneNode *parent) : SceneTreeNode(parent) { id = 0; }

void AbstractLight::updateShaderData(Shader *shader_program, glm::mat4 &modelview, unsigned int index) {
  if (shader_program) {
    std::string struct_name = light_struct_name + std::string("[") + std::to_string(index) + std::string("].");
    shader_program->setUniform(struct_name + std::string(uniform_name_vec3_lighting_position), viewspace_position);
    shader_program->setUniform(struct_name + std::string(uniform_name_float_lighting_intensity), intensity);
    shader_program->setUniform(struct_name + std::string(uniform_name_vec3_lighting_specular_color), specularColor);
    shader_program->setUniform(struct_name + std::string(uniform_name_vec3_lighting_diffuse_color), diffuseColor);
    shader_program->setUniform(struct_name + std::string(uniform_name_vec3_lighting_ambient_color), ambientColor);
  }
}

void AbstractLight::updateLightData(const LightData &data) {
  if (data.checkUpdateFlag(LightData::POSITION_UPDATE))
    setPosition(data.position);
  if (data.checkUpdateFlag(LightData::AMBIANT_UPDATE))
    setAmbiantColor(data.ambiant_col);
  if (data.checkUpdateFlag(LightData::DIFFUSE_UPDATE))
    setDiffuseColor(data.diffuse_col);
  if (data.checkUpdateFlag(LightData::SPECULAR_UPDATE))
    setSpecularColor(data.specular_col);
  if (data.checkUpdateFlag(LightData::INTENSITY_UPDATE))
    setIntensity(data.intensity);
}

/*****************************************************************************************************************/
/* Here , position is actually the direction ... */
DirectionalLight::DirectionalLight(ISceneNode *parent) : AbstractLight(parent) {
  position = glm::vec3(0.f);
  type = DIRECTIONAL;
  intensity = 0.f;
  specularColor = glm::vec3(1.f);
  ambientColor = glm::vec3(1.f);
  diffuseColor = glm::vec3(1.f);
  light_struct_name = std::string(uniform_name_str_lighting_directional_struct_name);
}

DirectionalLight::DirectionalLight(
    glm::vec3 _position, glm::vec3 _ambientColor, glm::vec3 _diffuseColor, glm::vec3 _specularColor, float _intensity, ISceneNode *parent)
    : DirectionalLight(parent) {
  specularColor = _specularColor;
  ambientColor = _ambientColor;
  diffuseColor = _diffuseColor;
  position = _position;
  intensity = _intensity;
  type = DIRECTIONAL;
}

DirectionalLight::DirectionalLight(glm::vec3 _position, glm::vec3 color, float _intensity, ISceneNode *parent) : DirectionalLight(parent) {
  position = _position;
  specularColor = color;
  diffuseColor = color;
  ambientColor = color;
  intensity = _intensity;
}

DirectionalLight::DirectionalLight(const LightData &data)
    : DirectionalLight(data.direction, data.ambiant_col, data.diffuse_col, data.specular_col, data.intensity, data.parent) {
  setName(data.name);
}

DirectionalLight::~DirectionalLight() {}

void DirectionalLight::updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index) {
  glm::mat4 modelview = view * computeFinalTransformation();
  viewspace_position = glm::vec3(modelview * glm::vec4(position, 0.f));
  AbstractLight::updateShaderData(shader, modelview, index);
}

void DirectionalLight::updateLightData(const LightData &data) {
  AbstractLight::updateLightData(data);
  if (data.checkUpdateFlag(LightData::DIRECTION_UPDATE))
    setPosition(data.direction);
}

/*****************************************************************************************************************/

PointLight::PointLight(ISceneNode *parent) : AbstractLight(parent) {
  position = glm::vec3(0.f);
  type = POINT;
  intensity = 0.f;
  specularColor = glm::vec3(1.f);
  ambientColor = glm::vec3(1.f);
  diffuseColor = glm::vec3(1.f);
  attenuation = glm::vec3(1.f);
  light_struct_name = std::string(uniform_name_str_lighting_point_struct_name);
}

PointLight::PointLight(glm::vec3 _position,
                       glm::vec3 _ambientColor,
                       glm::vec3 _diffuseColor,
                       glm::vec3 _specularColor,
                       glm::vec3 _attenuation,
                       float _intensity,
                       ISceneNode *parent)
    : PointLight(parent) {
  specularColor = _specularColor;
  ambientColor = _ambientColor;
  diffuseColor = _diffuseColor;
  position = _position;
  intensity = _intensity;
  type = POINT;
  attenuation = _attenuation;
}

PointLight::PointLight(glm::vec3 _position, glm::vec3 color, glm::vec3 _attenuation, float _intensity, ISceneNode *parent) : PointLight(parent) {
  position = _position;
  specularColor = color;
  diffuseColor = color;
  ambientColor = color;
  intensity = _intensity;
  attenuation = _attenuation;
}

PointLight::PointLight(const LightData &data)
    : PointLight(data.position, data.ambiant_col, data.diffuse_col, data.specular_col, data.attenuation, data.intensity, data.parent) {
  setName(data.name);
}

PointLight::~PointLight() {}

void PointLight::updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index) {
  local_transformation = glm::translate(glm::mat4(1.f), position);
  glm::mat4 modelview = view * computeFinalTransformation();
  viewspace_position = glm::vec3(modelview * glm::vec4(position, 1.f));
  if (shader) {
    std::string struct_name = light_struct_name + std::string("[") + std::to_string(index) + std::string("].");
    shader->setUniform(struct_name + uniform_name_float_lighting_attenuation_constant, attenuation.x);
    shader->setUniform(struct_name + uniform_name_float_lighting_attenuation_linear, attenuation.y);
    shader->setUniform(struct_name + uniform_name_float_lighting_attenuation_quadratic, attenuation.z);
  }
  AbstractLight::updateShaderData(shader, modelview, index);
}

void PointLight::updateLightData(const LightData &data) {
  AbstractLight::updateLightData(data);
  if (data.checkUpdateFlag(LightData::ATTENUATION_UPDATE))
    setAttenuation(data.attenuation);
}

/*****************************************************************************************************************/

SpotLight::SpotLight(ISceneNode *parent) : AbstractLight(parent) {
  position = glm::vec3(0.f);
  type = SPOT;
  theta = 0.f;
  intensity = 0.f;
  specularColor = glm::vec3(1.f);
  ambientColor = glm::vec3(1.f);
  diffuseColor = glm::vec3(1.f);
  direction = glm::vec3(0.f);
  light_struct_name = std::string(uniform_name_str_lighting_spot_struct_name);
}

SpotLight::SpotLight(glm::vec3 _position, glm::vec3 _direction, glm::vec3 _color, float _cutoff_angle, float _intensity, ISceneNode *parent)
    : SpotLight(parent) {
  position = _position;
  specularColor = _color;
  diffuseColor = _color;
  ambientColor = _color;
  intensity = _intensity;
  theta = _cutoff_angle;
  direction = _direction;
}

SpotLight::SpotLight(glm::vec3 _position,
                     glm::vec3 _direction,
                     glm::vec3 _ambient,
                     glm::vec3 _diffuse,
                     glm::vec3 _specular,
                     float _angle,
                     float _intensity,
                     ISceneNode *parent)
    : SpotLight(parent) {
  specularColor = _specular;
  ambientColor = _ambient;
  diffuseColor = _diffuse;
  position = _position;
  intensity = _intensity;
  theta = _angle;
  direction = _direction;
}

SpotLight::SpotLight(const LightData &data)
    : SpotLight(data.position, data.direction, data.ambiant_col, data.diffuse_col, data.specular_col, data.theta, data.intensity, data.parent) {
  setName(data.name);
}

void SpotLight::updateShaderData(Shader *shader, glm::mat4 &view, unsigned int index) {
  local_transformation = glm::translate(glm::mat4(1.f), position);
  glm::mat4 modelview = view * computeFinalTransformation();
  viewspace_position = glm::vec3(modelview * glm::vec4(position, 1.f));
  viewspace_direction = glm::vec3(modelview * glm::vec4(direction - position, 0.f));
  float rad_theta = glm::radians(theta);
  if (shader) {
    std::string struct_name = light_struct_name + std::string("[") + std::to_string(index) + std::string("].");
    shader->setUniform(struct_name + std::string(uniform_name_vec3_lighting_spot_direction), viewspace_direction);
    shader->setUniform(struct_name + std::string(uniform_name_float_lighting_spot_theta), rad_theta);
  }
  AbstractLight::updateShaderData(shader, modelview, index);
}

void SpotLight::updateLightData(const LightData &data) {
  AbstractLight::updateLightData(data);
  if (data.checkUpdateFlag(LightData::DIRECTION_UPDATE))
    setDirection(data.direction);
  if (data.checkUpdateFlag(LightData::THETA_UPDATE))
    setAngle(data.theta);
}
