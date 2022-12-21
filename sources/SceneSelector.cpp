#include "../includes/SceneSelector.h"

using namespace axomae ;



SceneSelector *SceneSelector::instance = nullptr;

SceneSelector* SceneSelector::getInstance(){
	if(instance == nullptr)
		instance = new SceneSelector();
	return instance ; 
}

void SceneSelector::remove(){
	if(instance != nullptr)
		delete instance ;
	instance = nullptr; 
}

SceneSelector::SceneSelector(){}
SceneSelector::~SceneSelector(){}

void SceneSelector::setScene(std::vector<Mesh> &meshes){
	scene = meshes ; 
	mesh_index = 0 ; 
}


Mesh SceneSelector::toNext(){
	if(scene.size() > 0){
		if((mesh_index + 1) >= scene.size()){
			mesh_index = 0 ; 
			return scene[mesh_index] ; 
		}
		else
			return scene[mesh_index ++] ;   
	}
	else
		return Mesh();
}

Mesh SceneSelector::toPrevious(){
	if(scene.size() > 0){	
		if(mesh_index == 0){
			mesh_index = scene.size() - 1 ; 
			return scene[mesh_index] ; 
		}
		else
			return scene[mesh_index --] ; 
	}
	else 
		return Mesh(); 
}

Mesh SceneSelector::getCurrent(){
	if(scene.size() > 0)
		return scene[mesh_index] ;
	else
		return Mesh(); 
}
