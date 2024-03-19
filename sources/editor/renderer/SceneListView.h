#ifndef SCENELISTVIEW_H
#define SCENELISTVIEW_H
#include <QTreeWidget>

#include "SceneHierarchy.h"

class NodeItem : public QTreeWidgetItem {
  friend class SceneListView;
  friend class NodeItemBuilder;

 public:
  std::string name;

 public:
  NodeItem(std::string n, int type, QTreeWidgetItem *parent = nullptr) : QTreeWidgetItem(parent, type) { name = n; }
  virtual void setItemText(int column) { QTreeWidgetItem::setText(column, QString(name.c_str())); }
};

class NodeItemBuilder {
 public:
  static NodeItem *buildNode(std::string name, int type, QTreeWidgetItem *parent = nullptr) { return new NodeItem(name, type, parent); }
};

// TODO: [AX-57] implement a visitor node for icon attributions to scene elements
class SceneListView : virtual public QTreeWidget {
 private:
  std::vector<NodeItem *> items;                  // Only used to keep track of elements in the tree.
  std::map<ISceneNode *, NodeItem *> node_lookup; /*<Keeps track of the NodeItems and their corresponding INodes*/
  SceneTree *current_scene;                       /*<Keeps track of the currently processed scene*/

 public:
  SceneListView(QWidget *parent = nullptr);
  virtual void emptyTree();
  INode *getSceneNode(const NodeItem *searched);
  NodeItem *getRoot() const;
  const NodeItem *getConstRoot() const;
  void updateSceneList();
  virtual void setScene(SceneTree &scene);
};

#endif