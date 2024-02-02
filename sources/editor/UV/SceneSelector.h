#ifndef SCENESELECTOR_H
#define SCENESELECTOR_H

#include "Loader.h"
#include "Mesh.h"

/**
 * @brief Keeps track of currently displayed mesh , and the parent 3D object
 */

class SceneSelector {
 public:
  static SceneSelector &getInstance();
  void setScene(std::vector<Mesh *> &meshes);
  void toNext();
  void toPrevious();
  Mesh *getCurrent();

 private:
  SceneSelector();
  std::vector<Mesh *> scene;
  unsigned int mesh_index;
  static SceneSelector *instance;
};

#endif