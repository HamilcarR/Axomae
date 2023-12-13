#include "../includes/SceneSelector.h"

using namespace axomae;

SceneSelector &SceneSelector::getInstance() {
  static SceneSelector instance;
  return instance;
}

SceneSelector::SceneSelector() {}

void SceneSelector::setScene(std::vector<Mesh *> &meshes) {
  scene = meshes;
  mesh_index = 0;
}

void SceneSelector::toNext() {
  if (scene.size() > 0) {
    if ((mesh_index + 1) >= scene.size())
      mesh_index = 0;
    else
      mesh_index++;
  }
}

void SceneSelector::toPrevious() {
  if (scene.size() > 0) {
    if (mesh_index == 0)
      mesh_index = scene.size() - 1;
    else
      mesh_index--;
  }
}

Mesh *SceneSelector::getCurrent() {
  if (scene.size() > 0)
    return scene[mesh_index] == nullptr ? nullptr : scene[mesh_index];
  else
    return nullptr;
}
