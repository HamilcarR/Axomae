#ifndef MESH_H
#define MESH_H

#include "utils_3D.h" 
#include "Material.h" 
#include "Camera.h"

/**
 * @brief Mesh.h
 * Mesh class implementation
 * 
 */
namespace axomae {

/**
 * @brief Mesh class
 * 
 */
class Mesh{
public:
	/**
	 * @brief Behavior of the depth function for the mesh
	 * 
	 */
	enum DEPTHFUNC: GLenum 
	{ 
		NEVER = GL_NEVER ,					 
		LESS = GL_LESS    , 
		EQUAL = GL_EQUAL  , 
		LESS_OR_EQUAL = GL_LEQUAL , 
		GREATER = GL_GREATER , 
		GREATER_OR_EQUAL = GL_GEQUAL , 
		NOT_EQUAL = GL_NOTEQUAL , 
		ALWAYS = GL_ALWAYS
	} ;  

	/**
	 * @brief Construct a new Mesh object
	 * 
	 */
	Mesh();
	
	/**
	 * @brief Copy constructor
	 * 
	 * @param copy To be copied
	 */
	Mesh(const Mesh& copy); 
	
	/**
	 * @brief Construct a new Mesh object from geometry and material
	 * 
	 * @param obj Geometry data
	 * @param mat Material data
	 */
	Mesh(Object3D const& obj , Material const& mat); 
	
	/**
	 * @brief Construct a new Mesh object with a name , geometry , and material
	 * 
	 * @param name String name
	 * @param obj Geometry data
	 * @param mat Material data
	 */
	Mesh(std::string name , Object3D const& obj , Material const& mat); 
	
	/**
	 * @brief Destroy the Mesh object
	 * 
	 */
	virtual ~Mesh();
	
	/**
	 * @brief Enable the mesh's material to render
	 * 
	 */
	virtual void bindMaterials();
	
	/**
	 * @brief Activates the current mesh's shader
	 * 
	 */
	virtual void bindShaders();
	
	/**
	 * @brief Releases the mesh's shader
	 * 
	 */
	virtual void releaseShaders(); 
	
	/**
	 * @brief Cleans all data of the mesh
	 * 
	 */
	virtual void clean() ; 
	
	/**
	 * @brief Check if mesh is initialized and ready to be rendererd 
	 * 
	 * @return true 
	 */
	virtual bool isInitialized();
	
	/**
	 * @brief Initialize OpenGL related data in the material and shaders of the mesh
	 * 
	 */
	virtual void initializeGlData(); 
	
	/**
	 * @brief Set the Scene Camera Pointer
	 * 
	 * @param camera Pointer on the scene camera
	 */
	virtual void setSceneCameraPointer(Camera *camera); 
	
	/**
	 * @brief Set the Rotation Matrix 
	 * 
	 * @param rot Rotation matrix
	 */
	virtual void setRotationMatrix(glm::mat4 &rot){ model_matrix = rot ; } //TODO : replace with setModelMatrix()
	
	/**
	 * @brief Disable the rendering of the back face of the mesh
	 * 
	 */
	void cullBackFace() ;
	
	/**
	 * @brief Disable the rendering of the front face 
	 * 
	 */
	void cullFrontFace() ; 
	
	/**
	 * @brief Disable both front and back face for rendering
	 * 
	 */
	void cullFrontAndBackFace() ; 
	
	/**
	 * @brief Enable face culling for this mesh
	 * 
	 * @param value True if we want to enable face culling
	 */
	void setFaceCulling(bool value) ; 	
	
	/**
	 * @brief Set the Depth  behavior of the mesh
	 * 
	 * @param value True if we want to enable depth functions
	 */
	void setDepthMask(bool value);
	
	/**
	 * @brief Set the Depth function to func
	 * 
	 * @param func Depth function to use. 
	 * @see DEPTHFUNC
	 */
	void setDepthFunc(DEPTHFUNC func); 
public:
	Object3D geometry;					/**<3D Geometry of the mesh , vertex positions , UVs etc*/
	Shader* shader_program; 			/**<Shader to be used for the mesh*/
	Material material; 					/**<Material to be used for the mesh*/
	std::string name; 					/**<Name of the mesh*/

protected:
	bool mesh_initialized ; 			/**<Is the mesh ready to render*/
	Camera* camera; 					/**<Pointer on the scene camera*/
	glm::mat4 model_matrix;				/**<Mesh's model matrix*/
	bool face_culling_enabled ;			/**<Is culling enabled*/
	bool depth_mask_enabled ; 			/**<Is depth enabled*/
};

/**
 * @brief Cubemap Mesh class
 * 
 */
class CubeMapMesh : public Mesh{
public:
	
	/**
	 * @brief Construct a new Cube Map Mesh object
	 * 
	 */
	CubeMapMesh(); 
	
	/**
	 * @brief Destroy the Cube Map Mesh object
	 * 
	 */
	virtual ~CubeMapMesh(); 
	
	/**
	 * @brief Bind the cubemap's shader
	 * 
	 */
	virtual void bindShaders(); 	
};







}
#endif
