#include "../includes/Scene.h"
#include "../includes/BoundingBox.h"

using namespace axomae ; 

Scene::Scene(){
    display_bbox = false; 
}

Scene::~Scene(){
    
}

void Scene::updateTree(){
    scene_tree.updateOwner(); 
    scene_tree.updateAccumulatedTransformations(); 
}

void Scene::setCameraPointer(Camera* _scene_camera){
    scene_camera = _scene_camera;
    scene_tree.pushNewRoot(scene_camera); 
}

void Scene::setScene(std::pair<std::vector<Mesh*> , SceneTree> &to_copy){
    scene_tree = to_copy.second; 
    for(auto A : to_copy.first){
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
            //glm::mat4 modelview_matrix = A->getMeshPointer()->computeFinalTransformation() * scene_camera->getView();
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
    scene_tree.clear(); 
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
    glm::mat4 view_matrix = scene_camera->getView(); 
    for(Drawable *A : meshes){	
		A->bind(); 		 
        light_database->updateShadersData(A->getMeshShaderPointer() , view_matrix); 
        glDrawElements(GL_TRIANGLES , A->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
		A->unbind();
	}	
}

//TODO: [AX-39] Merge bounding box geometry 
void Scene::drawBoundingBoxes(){
    if(display_bbox){
        std::vector<Drawable*> bounding_boxes = getBoundingBoxElements(); 
        for(Drawable* A : bounding_boxes){
            A->bind();
            glDrawElements(GL_TRIANGLES , A->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
            A->unbind(); 
        }
    }
}

std::vector<INode*> Scene::getNodeByName(const std::string& name){
    return scene_tree.findByName(name); 
}

void Scene::setPolygonWireframe(){
    for(auto A : scene){
        Mesh* tmp = A.drawable->getMeshPointer(); 
        if(tmp)
            tmp->setPolygonDrawMode(Mesh::LINE); 
    }
}

void Scene::setPolygonPoint(){
    for(auto A : scene){
        Mesh* tmp = A.drawable->getMeshPointer(); 
        if(tmp)
            tmp->setPolygonDrawMode(Mesh::POINT); 
    }
}

void Scene::setPolygonFill(){
    for(auto A : scene){
        Mesh* tmp = A.drawable->getMeshPointer(); 
        if(tmp)
            tmp->setPolygonDrawMode(Mesh::FILL); 
    }
}



