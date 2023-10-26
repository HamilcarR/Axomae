#ifndef SCENELISTVIEW_H
#define SCENELISTVIEW_H
#include <QTreeWidget>

#include "SceneHierarchy.h"



class NodeItem : virtual public QTreeWidgetItem {
    friend class SceneListView;
public : 
    NodeItem(std::string n , int type) : QTreeWidgetItem(type) , name(n){}
    virtual ~NodeItem(){

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

    virtual void setScene(const SceneTree& scene){
       const INode* root_node = scene.getRootNode(); 
       auto layout_nodes_lambda = [](const ISceneNode* node , const SceneTree& scene , SceneListView& scene_view_list , std::vector<NodeItem*> &r_items){
            if(node == scene.getRootNode()){
                NodeItem *root = new NodeItem(node->getName() , QTreeWidgetItem::Type);
                root->setItemText(0);
                scene_view_list.addTopLevelItem(root);  
                r_items.push_back(root); 
            }
            else{
                NodeItem *current = new NodeItem(node->getName() , QTreeWidgetItem::Type);
                current->setItemText(0); 
                
            }
       };
       
       
       /*NodeItem *root1 = new NodeItem(std::string("root1") , QTreeWidgetItem::Type);
       NodeItem *child = new NodeItem(std::string("child") , QTreeWidgetItem::Type); 
       NodeItem *root2 = new NodeItem(std::string("root2") , QTreeWidgetItem::Type); 
       
       root1->setItemText(0);
       root2->setItemText(0); 
       child->setItemText(0);
       root1->addChild(child); 
       addTopLevelItem(root1);
       addTopLevelItem(root2);*/   
    } 

private:
    std::vector<NodeItem*> items ; 
};

















#endif