#include "../includes/SceneSelector.h"

using namespace axomae ;

SceneSelector::SceneSelector(std::vector<Mesh> &meshes){
	scene = meshes ; 
	mesh_index = 0 ; 
}

SceneSelector::~SceneSelector(){
	
}

Mesh SceneSelector::getNext(){
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

Mesh SceneSelector::getPrevious(){
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
