#include "SceneSelector.h"

SceneSelector::SceneSelector() {}

void SceneSelector::setScene(std::vector<Mesh *> &meshes) {
  scene = meshes;
  current_mesh_index = 0;
}

void SceneSelector::toNext() {
  if (scene.size() > 0) {
    if (static_cast<size_t>((current_mesh_index + 1)) >= scene.size())
      current_mesh_index = 0;
    else
      current_mesh_index++;
  }
}

void SceneSelector::toPrevious() {
  if (scene.size() > 0) {
    if (current_mesh_index == 0)
      current_mesh_index = scene.size() - 1;
    else
      current_mesh_index--;
  }
}

Mesh *SceneSelector::getCurrent() {
  if (scene.size() > 0)
    return scene[current_mesh_index] == nullptr ? nullptr : scene[current_mesh_index];
  else
    return nullptr;
}

bool SceneSelector::setCurrent(int new_current_mesh) {
  if (static_cast<size_t>(new_current_mesh) >= scene.size() || new_current_mesh < 0)
    return false;
  current_mesh_index = new_current_mesh;
  return true;
}