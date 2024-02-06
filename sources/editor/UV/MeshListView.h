#ifndef MESHLISTVIEW_H
#define MESHLISTVIEW_H

#include "constants.h"
#include <QListWidget>

/**
 * @file MeshListView.h
 * This file implements the list of meshes in the UV renderer UI
 *
 */

class Mesh;
class SceneSelector;
/**
 * @class MeshListView
 *
 */
class MeshListView : public QListWidget {
 public:
  MeshListView(QWidget *parent = nullptr);
  void setList(const std::vector<Mesh *> &meshes);
  void setSceneSelector(SceneSelector *scene_selector);
  void setSelected(int row);

 private:
  std::vector<std::unique_ptr<QListWidgetItem>> mesh_names_list;
  SceneSelector *uv_editor_mesh_selection{};
};

#endif
