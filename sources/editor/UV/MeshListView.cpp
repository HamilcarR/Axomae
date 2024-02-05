#include "MeshListView.h"
#include "Mesh.h"

using namespace axomae;

MeshListView::MeshListView(QWidget *parent) : QListWidget(parent) {}

void MeshListView::setList(const std::vector<Mesh *> &meshes) {
  mesh_names_list.clear();
  for (Mesh *A : meshes)
    mesh_names_list.push_back(std::make_unique<QListWidgetItem>(QString(A->getMeshName().c_str()), this, 0));
}
