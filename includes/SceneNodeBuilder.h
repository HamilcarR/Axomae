#ifndef SCENENODEBUILDER_H
#define SCENENODEBUILDER_H
#include "SceneHierarchy.h"
#include "Node.h"
#include "Mesh.h"
#include "LightingSystem.h"
#include "Camera.h"



/**
 * @file SceneNodeBuilder.h
 * This file implements a node builder 
 * 
 */


/**
 * @class SceneNodeBuilder
 * 
 */
class SceneNodeBuilder {
public: 
    
    template<class... Args>
    static SceneNodeInterface* buildCamera(SceneNodeInterface* parent, Args&&... args){
        return new Camera(std::forward<Args>(args)... , parent);
    } 
    
    template<class... Args>
    static SceneNodeInterface* buildEmptyNode(SceneNodeInterface* parent, Args&&... args){
        return new SceneTreeNode(std::forward<Args>(args)... , parent);
    } 

    template<class... Args>
    static SceneNodeInterface* buildMesh(SceneNodeInterface* parent, Args&&... args){
        return new Mesh(std::forward<Args>(args)... , parent);
    } 

    template<class... Args>
    static SceneNodeInterface* buildLight(AbstractLight::TYPE type , SceneNodeInterface* parent, Args&&... args){
        switch(type){
            case AbstractLight::POINT:
                return new PointLight(std::forward<Args>(args)... , parent); 
            break;
            case AbstractLight::DIRECTIONAL:
                return new DirectionalLight(std::forward<Args>(args)... , parent);
            break;
            case AbstractLight::SPOT:
                return new SpotLight(std::forward<Args>(args)... , parent);  
            default:
                return nullptr; 
            break; 
        }
            
    } 



    

};





#endif