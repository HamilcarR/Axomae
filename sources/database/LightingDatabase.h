#ifndef LIGHTINGDATABASE_H
#define LIGHTINGDATABASE_H
#include "INodeDatabase.h"
#include "LightingSystem.h"

class LightingDatabase {
  using AbstractLightMap = std::map<AbstractLight::TYPE, std::vector<AbstractLight *>>;

 protected:
  AbstractLightMap light_database; /**<Map of all lights in the scene. The map key stored is the type of light*/
  INodeDatabase *node_database;
  std::vector<unsigned int> free_id_list; /**<List of free IDs belonging to removed lights*/
  unsigned int last_id;

 public:
  LightingDatabase();
  virtual bool addLight(AbstractLight *light);
  virtual bool addLight(int database_index);
  virtual bool removeLight(AbstractLight *light);
  virtual bool removeLight(unsigned id);
  virtual bool updateLight(unsigned id, const LightData &data);
  ax_no_discard const std::vector<AbstractLight *> &getLightsArrayByType(AbstractLight::TYPE type) const;
  virtual bool contains(AbstractLight *light) const;
  void eraseLightsArray(AbstractLight::TYPE type);
  void clearDatabase();
  virtual void updateShadersData(Shader *shader, glm::mat4 &view);
  virtual void updateShadersData(AbstractLight::TYPE type, Shader *shader, glm::mat4 &view);
  AbstractLight *getLightFromID(unsigned id) const;

 protected:
  void giveID(AbstractLight *light);
};

#endif