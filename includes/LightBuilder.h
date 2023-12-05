#ifndef LIGHTBUILDER_H
#define LIGHTBUILDER_H
#include "LightingSystem.h"
#include "SceneHierarchy.h"

class LightBuilder {
 public:
  /**
   * @brief Create a point light
   *
   * @param light_data Data of the light . Will take into account only the fields used by point lights .
   * @param ancestor Parent of the light node
   */
  static AbstractLight *createPLight(const LightData &light_data);

  /**
   * @brief Create a directional light
   *
   * @param light_data Data of the light
   * @param ancestor Parent of the light node
   */
  static AbstractLight *createDLight(const LightData &light_data);

  /**
   * @brief Create a spot light
   *
   * @param light_data Data of the light
   * @param ancestor Parent of the light node
   */
  static AbstractLight *createSLight(const LightData &light_data);

  /**
   * @brief Create an area light
   *
   * @param light_data Data of the light
   * @param ancestor Parent of the light node
   */
  static AbstractLight *createALight(const LightData &light_data);
};

#endif