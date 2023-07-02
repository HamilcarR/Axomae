#ifndef NODE_H
#define NODE_H
#include "utils_3D.h"
#include <memory> 

class SceneNodeInterface {
public:

    virtual glm::mat4 getWorldModelMatrix() const = 0 ; 
    virtual const glm::mat4& getLocalModelMatrix() const {return local_modelmatrix;} 
    virtual void resetLocalModelMatrix(); 
    virtual const std::vector<SceneNodeInterface*>& getNodeChildren() const {return children ;}
    virtual SceneNodeInterface* getNodeParent() const {return parent;}
    virtual void addChildNode(SceneNodeInterface* node); 
    virtual void setParentNode(SceneNodeInterface* node); 

protected:
    glm::mat4 local_modelmatrix;
    SceneNodeInterface* parent ; 
    std::vector<SceneNodeInterface*> children ; 

};


class SceneNode : public SceneNodeInterface{
public:
    SceneNode(SceneNodeInterface* parent = nullptr);
    SceneNode(const SceneNode& copy); 
    virtual ~SceneNode();
    virtual glm::mat4 getWorldModelMatrix() const override ;
    virtual SceneNode& operator=(const SceneNode& copy);

protected:
};




#endif 