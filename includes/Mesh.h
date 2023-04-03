#ifndef MESH_H
#define MESH_H

#include "utils_3D.h" 
#include "Material.h" 
#include "Camera.h"

namespace axomae {


class Mesh{
public:
	Mesh();
	Mesh(const Mesh& copy); 
	Mesh(Object3D const& obj , Material const& mat); 
	Mesh(std::string name , Object3D const& obj , Material const& mat); 
	virtual ~Mesh();
	virtual void bindMaterials();
	virtual void bindShaders();
	virtual void releaseShaders(); 
	virtual void clean() ; 
	virtual bool isInitialized();
	virtual void initializeGlData(); 
	virtual void setSceneCameraPointer(Camera *camera); 

public:
	Object3D geometry;
	Shader shader_program; 
	Material material; 
	std::string name; 

private:
	bool mesh_initialized ; 
	Camera* camera; 











};










}
#endif
