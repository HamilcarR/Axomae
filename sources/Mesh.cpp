#include "../includes/Mesh.h"


namespace axomae {

Mesh::Mesh(){
	mesh_initialized = false ; 
}

Mesh::Mesh(const Mesh& copy){
	geometry = copy.geometry ; 
	material = copy.material ; 
	name = copy.name ; 

}

Mesh::Mesh(Object3D const& geo , Material const& mat){
	geometry = geo; 
	material = mat;
}

Mesh::Mesh(std::string n , Object3D const& geo , Material const& mat){
	geometry = geo; 
	material = mat;
	name = n ; 
}

Mesh::~Mesh(){

}

void Mesh::initializeGlData(){
	shader_program.initializeShader();
	material.initializeMaterial();
	mesh_initialized = true ; 
}

void Mesh::bindMaterials(){
	material.bind(); 

}

void Mesh::clean(){
	shader_program.clean(); 
	material.clean(); 
}

bool Mesh::isInitialized(){
	return mesh_initialized; 	
}

























}//end namespace
