#include "LightingDatabase.h"
#include "ResourceDatabaseManager.h"
#include "UniformNames.h"

/* Sets up a new ID for the light if none is available , or give an old one that belonged to a former light*/
void LightingDatabase::giveID(AbstractLight *light) {
  if (free_id_list.empty())
    light->setID(last_id++);
  else {
    auto it = free_id_list.begin();
    light->setID(*it);
    free_id_list.erase(it);
  }
}

LightingDatabase::LightingDatabase() {
  last_id = 0;
  node_database = ResourceDatabaseManager::getInstance().getNodeDatabase();
}

bool LightingDatabase::addLight(int node_database_index) {
  AbstractLight *light = dynamic_cast<AbstractLight *>(node_database->get(node_database_index));
  assert(light);
  if (contains(light))
    return false;
  AbstractLight::TYPE type = light->getType();
  light_database[type].push_back(light);
  giveID(light);
  return true;
}

bool LightingDatabase::addLight(AbstractLight *light) {
  assert(light);
  if (contains(light))
    return false;
  AbstractLight::TYPE type = light->getType();
  light_database[type].push_back(light);
  giveID(light);
  return true;
}

bool LightingDatabase::removeLight(AbstractLight *light) {
  assert(light != nullptr);
  for (auto array : light_database)
    for (auto it = array.second.begin(); it != array.second.end(); it++) {
      if (*it == light) {
        array.second.erase(it);
        node_database->remove(static_cast<NodeInterface *>(*it));
        return true;
      }
    }
  return false;
}

bool LightingDatabase::removeLight(const unsigned index) {
  for (auto array : light_database)
    for (auto it = array.second.begin(); it != array.second.end(); it++) {
      if ((*it)->getID() == index) {
        AbstractLight *temp = *it;
        array.second.erase(it);
        node_database->remove(static_cast<NodeInterface *>(*it));
        return true;
      }
    }
  return false;
}

AbstractLight *LightingDatabase::getLightFromID(const unsigned id) const {
  for (auto array : light_database)
    for (auto it = array.second.begin(); it != array.second.end(); it++)
      if ((*it)->getID() == id)
        return *it;
  return nullptr;
}

bool LightingDatabase::updateLight(const unsigned id, const LightData &data) {
  AbstractLight *l = getLightFromID(id);
  if (!l)
    return false;
  l->updateLightData(data);
  return true;
}

const std::vector<AbstractLight *> &LightingDatabase::getLightsArrayByType(AbstractLight::TYPE type) const { return light_database.at(type); }

void LightingDatabase::eraseLightsArray(AbstractLight::TYPE type) { light_database.erase(type); }

void LightingDatabase::clearDatabase() { light_database.clear(); }

void LightingDatabase::updateShadersData(AbstractLight::TYPE type, Shader *shader, glm::mat4 &view) {
  std::vector<AbstractLight *> array = light_database[type];
  unsigned int index = 0;
  for (auto it = array.begin(); it != array.end(); it++, index++)
    (*it)->updateShaderData(shader, view, index);
}

// First loop is to set up the max number of lights of each type.
void LightingDatabase::updateShadersData(Shader *shader, glm::mat4 &view) {
  for (auto it = light_database.begin(); it != light_database.end(); it++) {
    unsigned int light_num = it->second.size();
    switch (it->first) {
      case AbstractLight::DIRECTIONAL:
        shader->setUniform(std::string(uniform_name_uint_lighting_directional_number_name), light_num);
        break;
      case AbstractLight::POINT:
        shader->setUniform(std::string(uniform_name_uint_lighting_point_number_name), light_num);
        break;
      case AbstractLight::SPOT:
        shader->setUniform(std::string(uniform_name_uint_lighting_spot_number_name), light_num);
        break;
      case AbstractLight::AMBIANT:
      case AbstractLight::HEMISPHERE:
      case AbstractLight::QUAD:
      case AbstractLight::AREA_TEXTURE:
      default:
        break;
    }
    updateShadersData(it->first, shader, view);
  }
}

bool LightingDatabase::contains(AbstractLight *light) const {
  for (const auto &A : light_database) {
    for (const AbstractLight *B : A.second)
      if (light == B)
        return true;
  }
  return false;
}