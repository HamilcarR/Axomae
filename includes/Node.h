#ifndef NODE_H
#define NODE_H
#include "utils_3D.h"
#include <memory> 


/**
 * @file Node.h
 * Implements the node class in the scene
 * 
 */




class SceneHierarchyInterface; 
/**
 * @class SceneNodeInterface
 * @brief Provides an interface for a scene node
 */
class SceneNodeInterface {
public:

    /**
     * @brief Destroy the Scene Node Interface object
     * 
     */
    virtual ~SceneNodeInterface(){}
    
    /**
     * @brief Compute the final model matrix , by multiplying the accumulated transformation with the local transformation
     * 
     * @return glm::mat4 
     */
    virtual glm::mat4 computeFinalTransformation() const = 0 ;

    /**
     * @brief Get the Local Model Matrix 
     * 
     * @return const glm::mat4 
     */
    virtual const glm::mat4 getLocalModelMatrix() const {return local_transformation;} 
    
    /**
     * @brief Set the Local Model Matrix object
     * 
     * @param matrix 
     */
    virtual void setLocalModelMatrix(const glm::mat4 matrix){local_transformation = matrix; }
    
    /**
     * @brief Sets the local transformation to identity 
     * 
     */
    virtual void resetLocalModelMatrix(); 
    
    /**
     * @brief Returns the array of children 
     * 
     * @return const std::vector<SceneNodeInterface*>& 
     */
    virtual const std::vector<SceneNodeInterface*>& getChildren() const = 0 ; 
    
    /**
     * @brief Returns the array of parents
     * 
     * @return const std::vector<SceneNodeInterface*>& 
     */
    virtual const std::vector<SceneNodeInterface*>& getParents() const = 0 ; 
    
    /**
     * @brief Add a child in the array of children 
     * 
     * @param node 
     */
    virtual void addChildNode(SceneNodeInterface* node) = 0; 
    
    /**
     * @brief Set up a new array of parents
     * 
     * @param parents 
     */
    virtual void setParents(std::vector<SceneNodeInterface*> &parents) = 0;
    
    /**
     * @brief Empty up the list of parents , setting the size of property "parents" to 0 
     * 
     */
    virtual void emptyParents() ;
    
    /**
     * @brief Empty up the list of children
     * 
     */
    virtual void emptyChildren();  
    
    /**
     * @brief Check if present node doesn't have successors
     * 
     */
    virtual bool isLeaf() const ;
    
    /**
     * @brief Mark the current node
     * 
     * @param marked 
     */
    void setMark(bool marked){mark = marked;}
    
    /**
     * @brief Check if the current node is marked 
     * 
     */
    bool isMarked(){return mark;}
    
    /**
     * @brief Set the Name of the node
     * 
     * @param str 
     */
    void setName(std::string str){name = str;}
    
    /**
     * @brief Get the Name of the node
     * 
     * @return const std::string& 
     */
    const std::string& getName() const {return name;}
    
    /**
     * @brief Specify the structure that owns this node
     * 
     * @param _owner 
     */
    void setHierarchyOwner(SceneHierarchyInterface* _owner){owner = _owner;}
    
    /**
     * @brief Returns a pointer on the structure that owns this node
     * 
     * @return SceneHierarchyInterface* 
     */
    SceneHierarchyInterface* getHierarchyOwner() const {return owner;}
    
    /**
     * @brief Check if node is root 
     * 
     * @return true 
     * @return false 
     */
    virtual bool isRoot() const; 
    
    /**
     * @brief If node isn't root , will travel the hierarchy and returns the root of the hierarchy 
     * 
     * @return SceneNodeInterface* 
     */
    virtual SceneNodeInterface* returnRoot() = 0; 
    
    /**
     * @brief Set the Accumulated Model Matrix 
     * 
     * @param matrix 
     */
    virtual void setAccumulatedModelMatrix(const glm::mat4 &matrix){accumulated_transformation = matrix; }
    
    /**
     * @brief Get the Accumulated Model Matrix 
     * 
     * @return const glm::mat4& 
     */
    virtual const glm::mat4& getAccumulatedModelMatrix() const {return accumulated_transformation;}
protected:
    glm::mat4 local_transformation;           /*<Local transformation of the node*/ 
    glm::mat4 accumulated_transformation;     /*<Matrix equal to all ancestors transformations*/ 
    bool mark;                                /*<Generic mark , for graph traversal*/
    std::string name;                         /*<Name of the node*/
    std::vector<SceneNodeInterface*> parents ; /*<List of parents*/ 
    std::vector<SceneNodeInterface*> children ; /*<List of children*/ 
    SceneHierarchyInterface* owner ;            /*<Structure owning this hierarchy*/

//TODO: [AX-37] Replace with adjacency list + pointer on one node that is a parent
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
    virtual glm::mat4 computeFinalTransformation() const override ;
    
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