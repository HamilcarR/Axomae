#include "../includes/Scene.h"
#include "../includes/BoundingBox.h"

using namespace axomae ; 

Scene::Scene(){

}

Scene::~Scene(){
    
}

void Scene::setScene(std::vector<Mesh*> &to_copy){
    for(auto A : to_copy){
        Scene::AABB mesh ; 
        mesh.aabb = BoundingBox(A->geometry.vertices);
        mesh.drawable = new Drawable(A) ; 
        scene.push_back(mesh);  

    }
}

std::vector<Drawable*> Scene::getOpaqueElements() const {
    std::vector<Drawable*> to_return ;
    Drawable* cubemap ;  
    for(auto aabb : scene){
        Drawable *A = aabb.drawable; 
        Material *mat = A->getMaterialPointer();
        if(!mat->isTransparent()) 
            to_return.push_back(A); 
    }
    return to_return; 
}

void Scene::sortTransparentElements(){
    for(auto bbox : scene){
        Material *A = bbox.drawable->getMaterialPointer(); 
        if(A->isTransparent()){
            glm::mat4 modelview_matrix = bbox.drawable->getMeshPointer()->getModelViewMatrix(); 
            glm::vec3 updated_aabb_center = bbox.aabb.computeModelViewPosition(modelview_matrix);
            float dist_to_camera = glm::length(updated_aabb_center);
            sorted_transparent_meshes[dist_to_camera] = bbox.drawable; 
        }
    } 
}

std::vector<Drawable*> Scene::getSortedTransparentElements(){
    std::vector<Drawable*> transparent_meshes ; 
    sorted_transparent_meshes.clear(); 
    sortTransparentElements();  
    for(std::map<float , Drawable*>::reverse_iterator it = sorted_transparent_meshes.rbegin() ; it != sorted_transparent_meshes.rend() ; it++)
        transparent_meshes.push_back(it->second); 
    return transparent_meshes; 
}

void Scene::clear(){
    for(unsigned int i = 0 ; i < scene.size() ; i++)
		if(scene[i].drawable != nullptr){
			scene[i].drawable->clean();
			delete scene[i].drawable; 
		}	
    scene.clear(); 
    sorted_transparent_meshes.clear(); 
}

bool Scene::isReady(){
    for(auto object : scene)
		if(!object.drawable->ready())
			return false;
    return true ; 
}

void Scene::prepare_draw(Camera* scene_camera){
    for(auto aabb : scene){
		aabb.drawable->setSceneCameraPointer(scene_camera); 
		aabb.drawable->startDraw(); 
	}
}