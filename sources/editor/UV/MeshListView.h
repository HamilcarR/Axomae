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
/**
 * @class MeshListView
 *
 */
class MeshListView : public QListWidget {
 public:
  MeshListView(QWidget *parent = nullptr);
  void setList(const std::vector<Mesh *> &meshes);

 private:
  std::vector<std::unique_ptr<QListWidgetItem>> mesh_names_list;
};

#endif
