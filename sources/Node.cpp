#include "../includes/Node.h"
#include <algorithm>


void ISceneNode::resetLocalModelMatrix(){
    local_transformation = glm::mat4(1.f); 
}

bool INode::isLeaf() const {
    return children.empty();
}

bool INode::isRoot() const {
    if(parents.empty())
        return true; 
    else{
        for(INode* A : parents)
            if(A != nullptr)
                return false;
        return true;
    }
}

void INode::emptyParents(){
    parents.clear(); 
}

void INode::emptyChildren(){
    children.clear(); 
}
/**************************************************************************************************************************************/

SceneTreeNode::SceneTreeNode(ISceneNode *_parent , ISceneHierarchy* _owner){
    if(_parent != nullptr){
        std::vector<INode*> ret = {_parent} ; 
        setParents(ret);
    }
    setHierarchyOwner(_owner);  
    local_transformation = accumulated_transformation = glm::mat4(1.f); 
    mark = false ;
    name = "Generic-Hierarchy-Node";
    updated = true ;  
}

SceneTreeNode::SceneTreeNode(const std::string& _name , const glm::mat4& transformation , ISceneNode *parent , ISceneHierarchy* owner ) : SceneTreeNode(parent , owner){
    name = _name ; 
    local_transformation = transformation ; 
}

SceneTreeNode::SceneTreeNode(const SceneTreeNode& copy){
    local_transformation = copy.getLocalModelMatrix(); 
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

/**
 * The function returns the root node of a scene tree by traversing up the parent nodes until the root
 * is reached.
 * 
 * @return a pointer to a SceneNodeInterface object, specifically the root node of the scene tree.
 */
ISceneNode* SceneTreeNode::returnRoot() {
    parents.erase(std::remove(parents.begin() , parents.end() , nullptr) , parents.end()); 
    SceneTreeNode* iterator = getParent(); 
    if(iterator == nullptr)
        return this; 
    while(iterator->getParent() != nullptr)
        iterator = iterator -> getParent(); 
    return iterator;
}

/**
 * The function computes the final transformation matrix for a scene tree node by multiplying the
 * accumulated transformation matrix with the local transformation matrix.
 * 
 * @return a glm::mat4, which is a 4x4 matrix representing a transformation.
 */
glm::mat4 SceneTreeNode::computeFinalTransformation() {
    if(parents.empty() || parents[0] == nullptr)
        return local_transformation;
    else
        return accumulated_transformation * local_transformation; 
    
}

SceneTreeNode& SceneTreeNode::operator=(const SceneTreeNode& copy){
    if(this != &copy){
        local_transformation = copy.getLocalModelMatrix(); 
        parents.clear(); 
        children.clear();
        parents.push_back(copy.getParent()); 
        for(auto const& A : copy.getChildren())
            children.push_back(A); 
    }
    return *this; 
}

void SceneTreeNode::setParents(std::vector<INode*> &nodes){
    if(!nodes.empty()){
        parents.clear(); //!check if leak ?  
        parents.push_back(nodes[0]); 
        if(parents[0] != nullptr)
            parents[0]->addChildNode(this); 
    }
}

void SceneTreeNode::setParent(INode* node){
    if(node != nullptr){
        std::vector<INode*> ret = {node}; 
        setParents(ret); 
    }
}

SceneTreeNode* SceneTreeNode::getParent() const {
    if(!parents.empty())
        return static_cast<SceneTreeNode*>(parents[0]);
    else
        return nullptr;  
}

void SceneTreeNode::addChildNode(INode* node){
    if(node){
        bool contains = std::find(children.begin() , children.end() , node) != children.end();  
        if(!contains){
            children.push_back(node);
            std::vector<INode*> ret = {this}; 
            node->setParents(ret); 
        }
    }
}


/**************************************************************************************************************************************/

