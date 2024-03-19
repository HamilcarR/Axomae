#include "SceneListView.h"

SceneListView::SceneListView(QWidget *parent) : QTreeWidget(parent) {}

void SceneListView::emptyTree() {
  clear();
  items.clear();
  node_lookup.clear();
}

INode *SceneListView::getSceneNode(const NodeItem *searched) {
  if (!searched)
    return nullptr;
  for (auto pair : node_lookup) {
    if (searched == pair.second)
      return pair.first;
  }
  return nullptr;
}

NodeItem *SceneListView::getRoot() const { return items.empty() ? nullptr : items[0]; }

const NodeItem *SceneListView::getConstRoot() const { return items.empty() ? nullptr : items[0]; }

void SceneListView::updateSceneList() { setScene(*current_scene); }

void SceneListView::setScene(SceneTree &scene) {
  emptyTree();
  current_scene = &scene;
  INode *root_node = scene.getRootNode();
  auto layout_nodes_lambda = [](INode *node,
                                SceneTree &scene,
                                SceneListView &scene_view_list,
                                std::vector<NodeItem *> &r_items,
                                std::map<ISceneNode *, NodeItem *> &equiv_table) {
    if (node == scene.getRootNode()) {
      NodeItem *root = new NodeItem(node->getName(), QTreeWidgetItem::Type);
      root->setItemText(0);
      scene_view_list.addTopLevelItem(root);
      r_items.push_back(root);
      std::pair<ISceneNode *, NodeItem *> node_treewidget_pair(static_cast<ISceneNode *>(node), root);
      equiv_table.insert(node_treewidget_pair);
    } else {
      ISceneNode *parent_inode = static_cast<ISceneNode *>(node->getParents()[0]);
      NodeItem *parent_nodeitem = equiv_table[parent_inode];
      NodeItem *current = new NodeItem(node->getName(), QTreeWidgetItem::Type, parent_nodeitem);
      current->setItemText(0);
      r_items.push_back(current);
      std::pair<ISceneNode *, NodeItem *> node_treewidget_pair(static_cast<ISceneNode *>(node), current);
      equiv_table.insert(node_treewidget_pair);
    }
  };
  scene.dfs(root_node, layout_nodes_lambda, scene, *this, items, node_lookup);
}