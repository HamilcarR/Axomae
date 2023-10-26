#ifndef SCENELISTVIEW_H
#define SCENELISTVIEW_H
#include <QTreeWidget>

#include "SceneHierarchy.h"



class NodeItem : public QTreeWidgetItem {
    friend class SceneListView;
public : 
    virtual ~NodeItem(){
    }

    NodeItem(std::string n , int type , QTreeWidgetItem* parent = nullptr) : QTreeWidgetItem(parent , type){
        name = n ; 
    } 

    virtual void setItemText(int column) {
        QTreeWidgetItem::setText(column , QString(name.c_str())); 
    }

public:
    std::string name ; 
};


class SceneListView : virtual public QTreeWidget{
public:
    SceneListView(QWidget* parent = nullptr) : QTreeWidget(parent)
    {
        
    }

    virtual ~SceneListView()
    {
    }

    virtual void clear(){
        for(QTreeWidgetItemIterator it(this) ; *it ; it++){
            QTreeWidgetItem *item = *it ; 
            delete item ; 
        } 
    }

    virtual void setScene(const SceneTree& scene){
       clear(); 
       const INode* root_node = scene.getRootNode(); 
       auto layout_nodes_lambda = [](const INode* node , 
                                    const SceneTree& scene , 
                                    SceneListView& scene_view_list , 
                                    std::vector<NodeItem*> &r_items , 
                                    std::map<const ISceneNode* ,  NodeItem*> &equiv_table)
        {
            if(node == scene.getRootNode()){
                NodeItem *root = new NodeItem(node->getName() , QTreeWidgetItem::Type);
                root->setItemText(0);
                scene_view_list.addTopLevelItem(root);  
                r_items.push_back(root);
                std::pair<const ISceneNode* , NodeItem*> node_treewidget_pair(static_cast<const ISceneNode*>(node) , root); 
                equiv_table.insert(node_treewidget_pair); 
            }
            else{
                
                const ISceneNode* parent_inode = static_cast<ISceneNode*>(node->getParents()[0]); 
                NodeItem* parent_nodeitem = equiv_table[parent_inode]; 
                NodeItem *current = new NodeItem(node->getName() , QTreeWidgetItem::Type , parent_nodeitem);
                current->setItemText(0);
                r_items.push_back(current);
                std::pair<const ISceneNode* , NodeItem*> node_treewidget_pair(static_cast<const ISceneNode*>(node), current);
                equiv_table.insert(node_treewidget_pair);  
            }
        };
        std::map<const ISceneNode* , NodeItem*> equiv_table;
        scene.dfs(root_node ,layout_nodes_lambda , scene , *this , items , equiv_table);   
    } 

private:
    std::vector<NodeItem*> items ; //Only used to simply keep track of elements in the tree.
};

















#endif