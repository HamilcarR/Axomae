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

void Scene::generateBoundingBoxes(Shader* box_shader){
    for(Scene::AABB scene_drawable : scene){
        Mesh* mesh = scene_drawable.drawable->getMeshPointer();
        BoundingBoxMesh* bbox_mesh = new BoundingBoxMesh(mesh , scene_drawable.aabb , box_shader);
        Drawable* bbox_drawable = new Drawable(bbox_mesh);
        bounding_boxes_array.push_back(bbox_drawable); 
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

std::vector<Drawable*> Scene::getSortedSceneByTransparency(){
    std::vector<Drawable*> to_return ;
    Drawable* cubemap ; 
    sorted_transparent_meshes.clear();  
    for(auto aabb : scene){
        Drawable *A = aabb.drawable; 
        Material *mat = A->getMaterialPointer();
        if(!mat->isTransparent()) 
            to_return.push_back(A); 
        else{
            glm::mat4 modelview_matrix = A->getMeshPointer()->getModelViewMatrix(); 
            glm::vec3 updated_aabb_center = aabb.aabb.computeModelViewPosition(modelview_matrix);
            float dist_to_camera = glm::length(updated_aabb_center);
            sorted_transparent_meshes[dist_to_camera] = A; 
        }
    }
    for(std::map<float , Drawable*>::reverse_iterator it = sorted_transparent_meshes.rbegin() ; it != sorted_transparent_meshes.rend() ; it++)
        to_return.push_back(it->second);
    return to_return ; 

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

std::vector<Drawable*> Scene::getBoundingBoxElements(){
    return bounding_boxes_array; 
}

void Scene::clear(){
    for(unsigned int i = 0 ; i < scene.size() ; i++)
		if(scene[i].drawable != nullptr){
			scene[i].drawable->clean();
			delete scene[i].drawable; 
		}	
    scene.clear();
    for(auto A : bounding_boxes_array){
        if(A){
            A->clean(); 
            delete A; 
        }
    } 
    bounding_boxes_array.clear(); 
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
    for(auto A : bounding_boxes_array){
        A->setSceneCameraPointer(scene_camera); 
        A->startDraw(); 
    }
}

void Scene::drawForwardTransparencyMode(){
    std::vector<Drawable*> meshes = getSortedSceneByTransparency();  	
    scene_camera->computeViewProjection();
    glm::mat4 default_modelview_matrix = scene_camera->getView() * scene_camera->getSceneModelMatrix();
    for(Drawable *A : meshes){	
		A->bind(); 		 
        light_database->updateShadersData(A->getMeshShaderPointer() , default_modelview_matrix); 
        glDrawElements(GL_TRIANGLES , A->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
		A->unbind();
	}	
}

void Scene::drawBoundingBoxes(){
    std::vector<Drawable*> bounding_boxes = getBoundingBoxElements(); 
    for(Drawable* A : bounding_boxes){
		A->bind();
		glDrawElements(GL_TRIANGLES , A->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
		A->unbind(); 
	}
}













