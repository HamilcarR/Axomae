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
    static ISceneNode* buildCamera(ISceneNode* parent, Args&&... args){
        return new Camera(std::forward<Args>(args)... , parent);
    } 
    
    template<class... Args>
    static ISceneNode* buildEmptyNode(ISceneNode* parent, Args&&... args){
        return new SceneTreeNode(std::forward<Args>(args)... , parent);
    } 

    template<class... Args>
    static ISceneNode* buildMesh(ISceneNode* parent, Args&&... args){
        return new Mesh(std::forward<Args>(args)... , parent);
    } 

    //! maybe delete ? can't see this becoming that useful 
    template<AbstractLight::TYPE type , class... Args>
    static ISceneNode* buildLight(ISceneNode* parent, Args&&... args){
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