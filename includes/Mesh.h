#ifndef MESH_H
#define MESH_H

#include "utils_3D.h" 
#include "Material.h" 
#include "Camera.h"

namespace axomae {


class Mesh{
public:
	enum DEPTHFUNC: GLenum { NEVER = GL_NEVER , 
				LESS = GL_LESS    , 
				EQUAL = GL_EQUAL  , 
				LESS_OR_EQUAL = GL_LEQUAL , 
				GREATER = GL_GREATER , 
				GREATER_OR_EQUAL = GL_GEQUAL , 
				NOT_EQUAL = GL_NOTEQUAL , 
				ALWAYS = GL_ALWAYS} ;  

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
	virtual void setRotationMatrix(glm::mat4 &rot){ model_matrix = rot ; }
	void cullBackFace() ;
	void cullFrontFace() ; 
	void cullFrontAndBackFace() ; 
	void setFaceCulling(bool value) ; 	
	void setDepthMask(bool value);
	void setDepthFunc(DEPTHFUNC func); 
public:
	Object3D geometry;
	Shader* shader_program; 
	Material material; 
	std::string name; 

protected:
	bool mesh_initialized ; 
	Camera* camera; 
	glm::mat4 model_matrix;
	bool face_culling_enabled ;
	bool depth_mask_enabled ; 
};


class CubeMapMesh : public Mesh{
public:
	CubeMapMesh(); 
	virtual ~CubeMapMesh(); 
	virtual void bindShaders(); 	
};







}
#endif
