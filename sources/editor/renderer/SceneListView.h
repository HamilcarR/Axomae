#ifndef SCENELISTVIEW_H
#define SCENELISTVIEW_H
#include <QTreeWidget>

#include "SceneHierarchy.h"

class NodeItem : public QTreeWidgetItem {
  friend class SceneListView;
  friend class NodeItemBuilder;

 public:
  NodeItem(std::string n, int type, QTreeWidgetItem *parent = nullptr) : QTreeWidgetItem(parent, type) { name = n; }

  virtual void setItemText(int column) { QTreeWidgetItem::setText(column, QString(name.c_str())); }

 public:
  std::string name;
};

class NodeItemBuilder {
 public:
  static NodeItem *buildNode(std::string name, int type, QTreeWidgetItem *parent = nullptr) { return new NodeItem(name, type, parent); }
};

// TODO: [AX-57] implement a visitor node for icon attributions to scene elements
class SceneListView : virtual public QTreeWidget {
 public:
  SceneListView(QWidget *parent = nullptr) : QTreeWidget(parent) {}

  virtual void emptyTree() {
    clear();
    items.clear();
    node_lookup.clear();
  }

  INode *getSceneNode(const NodeItem *searched) {
    if (!searched)
      return nullptr;
    for (auto pair : node_lookup) {
      if (searched == pair.second)
        return pair.first;
    }
    return nullptr;
  }

  NodeItem *getRoot() const { return items.empty() ? nullptr : items[0]; }
  const NodeItem *getConstRoot() const { return items.empty() ? nullptr : items[0]; }

  void updateSceneList() { setScene(*current_scene); }

  virtual void setScene(SceneTree &scene) {
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

 private:
  std::vector<NodeItem *> items;                  // Only used to keep track of elements in the tree.
  std::map<ISceneNode *, NodeItem *> node_lookup; /*<Keeps track of the NodeItems and their corresponding INodes*/
  SceneTree *current_scene;                       /*<Keeps track of the currently processed scene*/
};

#endif