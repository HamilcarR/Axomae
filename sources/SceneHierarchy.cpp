#include "../includes/SceneHierarchy.h"





void SceneHierarchyInterface::setAsRootChild(SceneNodeInterface* node){
    if(root != nullptr)
        root->addChildNode(node); 
}

void SceneHierarchyInterface::updateOwner(){
    if(root != nullptr){
        auto scene_update_owner = [](SceneNodeInterface* node , SceneHierarchyInterface* owner){
            node->setHierarchyOwner(owner); 
        };
        dfs(root , scene_update_owner , this); 
    }
}

void SceneHierarchyInterface::clear(){
    for(auto A : generic_nodes_to_delete){
        delete A ; 
    }
    generic_nodes_to_delete.clear(); 
}

/*******************************************************************************************************************************************************************/

SceneTree::SceneTree(SceneNodeInterface* node){
    root = node;
}

SceneTree::SceneTree(const SceneTree& copy){
    root = copy.getRootNode();
    setIterator(copy.getIterator()); 
}

SceneTree::~SceneTree(){
}

void SceneTree::createGenericRootNode(){
    root = new SceneTreeNode(); 
}

SceneTree& SceneTree::operator=(const SceneTree& copy){
    if(this != &copy){
        root = copy.getRootNode(); 
        setIterator(copy.getIterator());
    }
    return *this ; 
}

void SceneTree::updateAccumulatedTransformations(){
    if(root != nullptr){
        auto recompute_matrices = [](SceneNodeInterface* node){
            if(!node->isRoot()){
                glm::mat4 new_accum = node->getParents()[0]->computeFinalTransformation();
                node->setAccumulatedModelMatrix(new_accum);
            }
            else
                node->setAccumulatedModelMatrix(glm::mat4(1.f));
        };
        dfs(root , recompute_matrices ); 
    }
}

void SceneTree::pushNewRoot(SceneNodeInterface* new_root){
    if(root != new_root){
        if(root == nullptr)
            root = new_root; 
        else{
            SceneNodeInterface* temp = root;
            new_root->emptyChildren();  
            new_root->addChildNode(root); 
            new_root->emptyParents(); 
            root = new_root ; 
            std::vector<SceneNodeInterface*> new_parent = {new_root}; 
            temp->setParents(new_parent); 
        }
        updateOwner() ; 
    }
}