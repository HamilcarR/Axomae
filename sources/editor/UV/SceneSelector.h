#ifndef SCENESELECTOR_H
#define SCENESELECTOR_H

#include "Loader.h"
#include "Mesh.h"

/**
 * @brief Keeps track of currently displayed mesh , and the parent 3D object
 */

class SceneSelector {
 public:
  SceneSelector();
  void setScene(std::vector<Mesh *> &meshes);
  void toNext();
  void toPrevious();
  Mesh *getCurrent();
  bool setCurrent(int new_current_mesh);
  int getCurrentId() { return current_mesh_index; }

 private:
  std::vector<Mesh *> scene{};
  int current_mesh_index{};
};

#endif
