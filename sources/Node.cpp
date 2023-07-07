#include "../includes/Node.h"
#include <algorithm>


void SceneNodeInterface::resetLocalModelMatrix(){
    local_modelmatrix = glm::mat4(1.f); 
}

bool SceneNodeInterface::isLeaf() const {
    return children.empty();
}

bool SceneNodeInterface::isRoot() const {
    if(parents.empty())
        return true; 
    else{
        for(SceneNodeInterface* A : parents)
            if(A != nullptr)
                return false;
        return true;
    }
}

void SceneNodeInterface::emptyParents(){
    parents.clear(); 
}

void SceneNodeInterface::emptyChildren(){
    children.clear(); 
}
/**************************************************************************************************************************************/

SceneTreeNode::SceneTreeNode(SceneNodeInterface *_parent , SceneHierarchyInterface* _owner){
    if(_parent != nullptr){
        std::vector<SceneNodeInterface*> ret = {_parent} ; 
        setParents(ret);
    }
    setHierarchyOwner(_owner);  
    local_modelmatrix = accumulated_modelmatrix = glm::mat4(1.f); 
    mark = false ;
    name = "Generic-Hierarchy-Node";  
}

SceneTreeNode::SceneTreeNode(const std::string& _name , const glm::mat4& transformation , SceneNodeInterface *parent , SceneHierarchyInterface* owner ) : SceneTreeNode(parent , owner){
    name = _name ; 
    local_modelmatrix = transformation ; 
}

SceneTreeNode::SceneTreeNode(const SceneTreeNode& copy){
    local_modelmatrix = copy.getLocalModelMatrix(); 
    parents.clear();  
    children.clear(); 
    parents.push_back(copy.getParents()[0]); 
    for(auto const &A : copy.getChildren()){
        children.push_back(A); 
    }
    setHierarchyOwner(copy.getHierarchyOwner()); 
}

SceneTreeNode::~SceneTreeNode(){

}

SceneNodeInterface* SceneTreeNode::returnRoot() {
    parents.erase(std::remove(parents.begin() , parents.end() , nullptr) , parents.end()); 
    SceneTreeNode* iterator = getParent(); 
    if(iterator == nullptr)
        return this; 
    while(iterator->getParent() != nullptr)
        iterator = iterator -> getParent(); 
    return iterator;
}

glm::mat4 SceneTreeNode::getWorldSpaceModelMatrix() const {
    if(parents.empty() || parents[0] == nullptr)
        return local_modelmatrix;
    else
        return accumulated_modelmatrix * local_modelmatrix; 
}

SceneTreeNode& SceneTreeNode::operator=(const SceneTreeNode& copy){
    if(this != &copy){
        local_modelmatrix = copy.getLocalModelMatrix(); 
        parents.clear(); 
        children.clear();
        parents.push_back(copy.getParent()); 
        for(auto const& A : copy.getChildren())
            children.push_back(A); 
    }
    return *this; 
}

void SceneTreeNode::setParents(std::vector<SceneNodeInterface*> &nodes){
    if(!nodes.empty()){
        parents.clear(); 
        parents.push_back(nodes[0]); 
        if(parents[0] != nullptr)
            parents[0]->addChildNode(this); 
    }
}

void SceneTreeNode::setParent(SceneNodeInterface* node){
    if(node != nullptr){
        std::vector<SceneNodeInterface*> ret = {node}; 
        setParents(ret); 
    }
}

SceneTreeNode* SceneTreeNode::getParent() const {
    if(!parents.empty())
        return static_cast<SceneTreeNode*>(parents[0]);
    else
        return nullptr;  
}

void SceneTreeNode::addChildNode(SceneNodeInterface* node){
    if(node){
        bool contains = std::find(children.begin() , children.end() , node) != children.end();  
        if(!contains){
            children.push_back(node);
            std::vector<SceneNodeInterface*> ret = {this}; 
            node->setParents(ret); 
        }
    }
}


/**************************************************************************************************************************************/

