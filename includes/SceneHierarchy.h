#ifndef SCENEHIERARCHY_H
#define SCENEHIERARCHY_H

#include <functional>
#include "Node.h"


/**
 * @file SceneGraph.h
 * This file implements the hierarchy system of a scene. 
 * 
 */

/**
 * @class SceneHierarchyInterface
 * 
 */
class SceneHierarchyInterface {
public:
    /**
     * @brief Set the Root object
     * 
     * @param _root 
     */
    void setRoot(SceneNodeInterface* _root){root = _root;}

    /**
     * @brief Returns the iterator of the structure 
     * 
     * @return SceneNodeInterface* 
     */
    SceneNodeInterface* getIterator() const {return iterator;}

    /**
     * @brief Set the Iterator of the structure
     * 
     * @param iter Iterator of type SceneNodeInterface*
     */
    void setIterator(SceneNodeInterface* iter){iterator = iter;} 
    
    /**
     * @brief Create an empty root node 
     * 
     */
    virtual void createGenericRootNode() = 0; 

    /**
     * @brief Get the Root Node object
     * 
     * @return SceneNodeInterface* 
     */
    SceneNodeInterface* getRootNode() const {return root;}

    /**
     * @brief Set the node as a child of root
     * 
     * @param node 
     */
    void setAsRootChild(SceneNodeInterface* node); 
    
    /**
     * @brief Traverse the scene structure, and set their owner of each node to this structure
     * 
     */
    virtual void updateOwner();
    
    /**
     * @brief Traverse the scene structure , and recomputes all transformations up to the accumulated_modelmatrix of each node
     * 
     */
    virtual void updateAccumulatedTransformations() = 0;

    /**
     * @brief Add node to the list of nodes that need to be deleted by the current structure. 
     * Specialized nodes like meshes , and lights , can be freed by their own scene structures. Empty generic nodes are deleted by the scene Tree/Graph structure.
     * 
     * @param node Node to track for deletion. 
     */
    virtual void addGenericNodeToDelete(SceneNodeInterface* node){generic_nodes_to_delete.push_back(node); }

    /**
     * @brief 
     * 
     */
    virtual void clear(); 
    /**
     * @brief 
     * 
     * @tparam Args 
     * @param begin 
     * @param func 
     * @param args 
     */
    template<class F , class ...Args>
    void dfs(SceneNodeInterface* begin ,F func, Args&& ...args);

private:
    
    /**
     * @brief 
     * 
     * @tparam Args 
     * @param node 
     * @param func 
     * @param args 
     */
    template<class F , class ...Args>
    void dfsTraverse(SceneNodeInterface* node ,F func , Args&& ...args); 



protected:
    SceneNodeInterface* root;
private:
    SceneNodeInterface* iterator;
    std::vector<SceneNodeInterface*> generic_nodes_to_delete; /*<Array of pointers on nodes that contain only a transformation*/  

};


/*******************************************************************************************************************************************************************/

class SceneTree : public SceneHierarchyInterface{
public:
    SceneTree(SceneNodeInterface* root = nullptr); 
    SceneTree(const SceneTree& copy);
    virtual void createGenericRootNode() override ;
    virtual SceneTree& operator=(const SceneTree& copy);
    virtual void updateAccumulatedTransformations() override;
    virtual void pushNewRoot(SceneNodeInterface* new_root); 
    virtual ~SceneTree(); 

private:


};













/*Templates definitions*/
/*******************************************************************************************************************************************************************/

template<class F , class ...Args>
void SceneHierarchyInterface::dfs(SceneNodeInterface* begin , F func , Args&& ...args){
    if((iterator = begin) != nullptr){
        dfsTraverse(iterator , func , std::forward<Args>(args)...); 
    }
}

template<class F , class ...Args>
void SceneHierarchyInterface::dfsTraverse(SceneNodeInterface* node , F func, Args&& ...args){
        if(node != nullptr){
            func(node , std::forward<Args>(args)...);
            for(SceneNodeInterface* child : node->getChildren())
                dfsTraverse(child , func , std::forward<Args>(args)...); 
        }
    }

#endif