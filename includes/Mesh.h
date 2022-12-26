#ifndef MESH_H
#define MESH_H

#include "utils_3D.h" 

namespace axomae {


class Mesh{
public:
	Mesh();
	Mesh(const Mesh& copy); 
	Mesh(Object3D const& obj , Material const& mat); 
	Mesh(std::string name , Object3D const& obj , Material const& mat); 
	virtual ~Mesh(); 


	Object3D geometry; 
	Material material; 
	std::string name; 













};










}
#endif
