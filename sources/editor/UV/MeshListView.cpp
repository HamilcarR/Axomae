#include "MeshListView.h"
#include "Mesh.h"
#include "SceneSelector.h"

using namespace axomae;

MeshListView::MeshListView(QWidget *parent) : QListWidget(parent) {
  /* Can only select one item */
  setSelectionMode(QAbstractItemView::SingleSelection);
}

void MeshListView::setList(const std::vector<Mesh *> &meshes) {
  mesh_names_list.clear();
  for (const Mesh *A : meshes)
    mesh_names_list.push_back(std::make_unique<QListWidgetItem>(QString(A->getMeshName().c_str()), this, 0));
}

void MeshListView::setSceneSelector(SceneSelector *scene_selector) { uv_editor_mesh_selection = scene_selector; }

void MeshListView::setSelected(int row) {
  QListWidgetItem *selected_item = item(row);
  if (selected_item) {
    setCurrentItem(selected_item);
  }
}
