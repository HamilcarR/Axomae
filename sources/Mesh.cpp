#include "../includes/Mesh.h"


namespace axomae {

Mesh::Mesh(){

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
