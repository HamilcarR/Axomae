#include "../includes/Node.h"
#include <algorithm>


void SceneNodeInterface::resetLocalModelMatrix(){
    local_modelmatrix = glm::mat4(1.f); 
}

void SceneNodeInterface::setParentNode(SceneNodeInterface* node){
    parent = node; 
    if(parent)
        parent->addChildNode(this); 
}

void SceneNodeInterface::addChildNode(SceneNodeInterface* node){
    if(node){
        bool contains = std::find(children.begin() , children.end() , node) != children.end();  
        if(!contains){
            children.push_back(node); 
            node->setParentNode(this); 
        }
    }
}

/**************************************************************************************************************************************/

SceneNode::SceneNode(SceneNodeInterface *_parent){
    setParentNode(_parent);  
    local_modelmatrix = glm::mat4(1.f); 
}

SceneNode::SceneNode(const SceneNode& copy){
    local_modelmatrix = copy.getLocalModelMatrix(); 
    parent = copy.getNodeParent(); 
    children = copy.getNodeChildren(); 
}

SceneNode::~SceneNode(){

}

glm::mat4 SceneNode::getWorldModelMatrix() const {
    if(parent == nullptr)
        return local_modelmatrix;
    else
        return parent->getWorldModelMatrix() * local_modelmatrix ; 
}

SceneNode& SceneNode::operator=(const SceneNode& copy){
    if(this != &copy){
        local_modelmatrix = copy.getLocalModelMatrix(); 
        parent = copy.getNodeParent(); 
        children = copy.getNodeChildren(); 
    }
    return *this; 
}