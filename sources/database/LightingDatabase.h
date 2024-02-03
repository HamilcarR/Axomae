#ifndef LIGHTINGDATABASE_H
#define LIGHTINGDATABASE_H
#include "INodeDatabase.h"
#include "LightingSystem.h"
#include "sources/common/math/utils_3D.h"

/**
 * @file LightingDatabase.h
 * Implements a database storing references to Lights .
 *
 */

/**
 * @class LightingDatabase
 * @note This class is non-owning. It only stores a pointer on an INode located in the Node database
 * @see INodeDatabase
 */
class LightingDatabase {
  using AbstractLightMap = std::map<AbstractLight::TYPE, std::vector<AbstractLight *>>;

 public:
  LightingDatabase();
  virtual ~LightingDatabase();

  virtual bool addLight(AbstractLight *light);
  virtual bool addLight(int database_index);
  virtual bool removeLight(AbstractLight *light);
  virtual bool removeLight(const unsigned id);
  virtual bool updateLight(const unsigned id, const LightData &data);
  const std::vector<AbstractLight *> &getLightsArrayByType(AbstractLight::TYPE type) const;
  virtual bool contains(AbstractLight *light) const;
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
  virtual void updateShadersData(Shader *shader, glm::mat4 &view);
  /**
   * @brief Update shader uniforms only for lights of "type"
   *
   * @param type Type of lights to send to the shader
   * @param shader Shader applied to the mesh
   * @param view View matrix for light transformations
   */
  virtual void updateShadersData(AbstractLight::TYPE type, Shader *shader, glm::mat4 &view);

  AbstractLight *getLightFromID(const unsigned id) const;

 protected:
  void giveID(AbstractLight *light);

 protected:
  AbstractLightMap light_database; /**<Map of all lights in the scene. The map key stored is the type of light*/
  INodeDatabase *node_database;
  std::vector<unsigned int> free_id_list; /**<List of free IDs belonging to removed lights*/
  unsigned int last_id;
};

#endif