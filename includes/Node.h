#ifndef NODE_H
#define NODE_H
#include "utils_3D.h"
#include <memory> 

class SceneHierarchyInterface; 
class SceneNodeInterface {
public:
    virtual ~SceneNodeInterface(){}
    virtual glm::mat4 getWorldSpaceModelMatrix() const = 0 ; 
    virtual const glm::mat4 getLocalModelMatrix() const {return local_modelmatrix;} 
    virtual void setLocalModelMatrix(const glm::mat4 matrix){local_modelmatrix = matrix; }
    virtual void resetLocalModelMatrix(); 
    virtual const std::vector<SceneNodeInterface*>& getChildren() const = 0 ; 
    virtual const std::vector<SceneNodeInterface*>& getParents() const = 0 ; 
    virtual void addChildNode(SceneNodeInterface* node) = 0; 
    virtual void setParents(std::vector<SceneNodeInterface*> &parents) = 0;
    virtual void emptyParents() ;
    virtual void emptyChildren();  
    virtual bool isLeaf() const ;
    void setMark(bool marked){mark = marked;}
    bool isMarked(){return mark;}
    void setName(std::string str){name = str;}
    const std::string& getName() const {return name;}
    void setHierarchyOwner(SceneHierarchyInterface* _owner){owner = _owner;}
    SceneHierarchyInterface* getHierarchyOwner() const {return owner;}
    virtual bool isRoot() const; 
    virtual SceneNodeInterface* returnRoot() = 0; 
    virtual void setAccumulatedModelMatrix(const glm::mat4 &matrix){accumulated_modelmatrix = matrix; }
    virtual const glm::mat4& getAccumulatedModelMatrix() const {return accumulated_modelmatrix;}
protected:
    glm::mat4 local_modelmatrix;
    glm::mat4 accumulated_modelmatrix; 
    bool mark;
    std::string name;  
    std::vector<SceneNodeInterface*> parents ;  
    std::vector<SceneNodeInterface*> children ; //TODO: [AX-37] Replace with adjacency list + pointer on one node that is a parent
    SceneHierarchyInterface* owner ;
};


/***************************************************************************************************************************************************/
/**
 * @class SceneTreeNode 
 * @brief Provides implementation for a scene tree node
 */
class SceneTreeNode : public SceneNodeInterface{
public:
    
    /**
     * @brief Construct a new Scene Tree Node object
     * 
     * @param parent Predecessor node in the scene hierarchy 
     * @param owner The structure that owns this node 
     */
    SceneTreeNode(SceneNodeInterface *parent = nullptr , SceneHierarchyInterface* owner = nullptr); 

    /**
     * @brief Construct a new Scene Tree Node object
     * 
     * @param name 
     * @param transformation
     * @param parent 
     * @param owner 
     */
    SceneTreeNode(const std::string& name , const glm::mat4& transformation , SceneNodeInterface *parent = nullptr , SceneHierarchyInterface* owner = nullptr); 

    /**
     * @brief Construct a new Scene Tree Node object
     * 
     * @param copy SceneTreeNode copy 
     */
    SceneTreeNode(const SceneTreeNode& copy); 
    
    /**
     * @brief Destroy the Node
     * 
     */
    virtual ~SceneTreeNode();
    
    /**
     * @brief Computes the final model matrix of the object in world space
     * 
     * @return glm::mat4 Model matrix in world space
     */
    virtual glm::mat4 getWorldSpaceModelMatrix() const override ;
    
    /**
     * @brief Assignment operator for the SceneTreeNode 
     * 
     * @param copy Copy to be assigned 
     * @return SceneTreeNode& Returns *this object 
     */
    virtual SceneTreeNode& operator=(const SceneTreeNode& copy);
    
    /**
     * @brief Add a child to the present node 
     * 
     * @param node 
     */
    virtual void addChildNode(SceneNodeInterface* node) override; 
    
    /**
     * @brief Set the parent of this node. 
     * 
     * @param node  
     */
    virtual void setParent(SceneNodeInterface* node) ;

    /**
     * @brief Get the Parent of this node
     * 
     * @return SceneNodeInterface* 
     */
    virtual SceneTreeNode* getParent() const ;  

    /**
     * @brief Get the children nodes collection of this present node
     * 
     * @return const std::vector<SceneNodeInterface*>& 
     */
    virtual const std::vector<SceneNodeInterface*>& getChildren() const {return children;} ; 
    
    /**
     * @brief Get the parents of this node . In case this is a SceneTreeNode , the returned vector is of size 1
     * 
     * @return std::vector<SceneNodeInterface*> 
     */
    virtual const std::vector<SceneNodeInterface*>& getParents() const {return parents ; }  

    /**
     * @brief Returns the root node of the tree
     * 
     * @return SceneNodeInterface* Root node
     */
    virtual SceneNodeInterface* returnRoot() ; 
protected:
    
    /**
    * @brief Set the predecessor of this node in the scene tree . Note that only the first element of the vector is stored as parent , 
    * as this structure is a tree. 
    * @param parents Vector of predecessors . Only the first element is considered
    */
    virtual void setParents(std::vector<SceneNodeInterface*> &parents) override;


};

/***************************************************************************************************************************************************/


































//TODO: [AX-35] Implement graph nodes


#endif 