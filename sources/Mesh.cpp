#include "../includes/Mesh.h"


namespace axomae {

Mesh::Mesh(){

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
































}//end namespace
