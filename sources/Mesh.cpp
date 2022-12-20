#include "../includes/Mesh.h"


namespace axomae {

Mesh::Mesh(){

}

Mesh::Mesh(Object3D const& geo , Material const& mat){
	geometry = geo; 
	material = mat;
}

Mesh::~Mesh(){

}
































}//end namespace
