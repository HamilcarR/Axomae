#ifndef MESH_H
#define MESH_H

#include "utils_3D.h" 
#include "Material.h" 
#include "Camera.h"
#include "BoundingBox.h"

/**
 * @brief Mesh.h
 * Mesh class implementation
 * 
 */

/**
 * @brief Mesh class
 * 
 */
class Mesh : public SceneTreeNode{
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
	 * @brief Rasterization mode for the drawn mesh
	 * 
	 */
	enum RASTERMODE : GLenum 
	{
		POINT = GL_POINT , 
		LINE = GL_LINE , 
		FILL = GL_FILL
	};

	
	/**
	 * @brief Construct a new Mesh , using parent as predecessor in the hierarchy
	 * 
	 * @param parent Parent object in the scene graph 
	 */
	Mesh(SceneNodeInterface* parent = nullptr);

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
	Mesh(const Object3D& obj , const Material& mat , SceneNodeInterface* parent = nullptr); 
	
	/**
	 * @brief Construct a new Mesh object with a name , geometry , and material
	 * 
	 * @param name String name
	 * @param obj Geometry data
	 * @param mat Material data
	 */
	Mesh(const std::string& name , const Object3D& obj , const Material& mat , SceneNodeInterface* parent = nullptr); 

	/**
	 * @brief Construct a new Mesh object
	 * 
	 * @param name Mesh name
	 * @param obj Mesh geometry
	 * @param mat Mesh material
	 * @param shader Shader used
	 */
	Mesh(const std::string& name , const Object3D& obj , const Material& mat , Shader* shader , SceneNodeInterface* parent = nullptr);

	/**
	 * @brief Construct a new Mesh object
	 * 
	 * @param name Mesh name 
	 * @param obj Mesh geometry
	 * @param mat Mesh material
	 * @param shader Shader used
	 */
	Mesh(const std::string& name , const Object3D&& obj , const Material& mat , Shader* shader , SceneNodeInterface* parent = nullptr); 
	
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
	 * @brief Disable the mesh's material to render
	 * 
	 */
	virtual void unbindMaterials(); 

	/**
	 * @brief Computes relevant matrices , and sets up the culling + depth states
	 * 
	 */
	virtual void preRenderSetup();

	/**
	 * @brief This method returns the GPU to the default rendering state.
	 * 
	 */
	virtual void afterRenderSetup(); 
	
	/**
	 * @brief This method does the shader data setup + binding 
	 * 
	 */
	virtual void setupAndBind();

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
	 * @brief Get the ModelView matrix
	 * 
	 * @return glm::mat4 
	 */
	virtual glm::mat4& getModelViewMatrix(){
		return modelview_matrix ; 
	}	
	
	/**
	 * @brief Control the rasterization mode
	 * 
	 */
	void setPolygonDrawMode(RASTERMODE mode); 

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

	/**
	 * @brief Get the Shader object
	 * 
	 * @return Shader* 
	 */
	Shader* getShader(){return shader_program;}

	/**
	 * @brief Set the Shader pointer in the mesh , and material
	 * 
	 * @param shader 
	 */
	void setShader(Shader* shader); 

	/**
	 * @brief Get the pointer on the current material properties
	 * 
	 * @return Material* 
	 */
	Material* getMaterial(){return &material;} 

	/**
	 * @brief Get the Mesh Name 
	 * 
	 * @return std::string 
	 */
	std::string getMeshName(){return name;}

	/**
	 * @brief Set the Mesh Name 
	 * 
	 * @param new_name 
	 */
	void setMeshName(std::string new_name){name = new_name;}

	/**
	 * @brief Get the geometry data of the mesh
	 * 
	 * @return Object3D 
	 */
	Object3D getGeometry(){return geometry;}

	/**
	 * @brief Set the Geometry object
	 * 
	 * @param _geometry 
	 */
	void setGeometry(Object3D _geometry){geometry = _geometry;}

	void setCubemapPointer(Mesh* cubemap_pointer){cubemap_reference = cubemap_pointer; }

	void setDrawState(bool draw){is_drawn = draw;}

	bool isDrawn(){return is_drawn;}

public:
	Object3D geometry;					/**<3D Geometry of the mesh , vertex positions , UVs etc*/	
	Material material; 					/**<Material to be used for the mesh*/
protected:
	bool mesh_initialized ; 			/**<Is the mesh ready to render*/
	Mesh* cubemap_reference ;			/**<Tracks the scene's cubemap*/ 
	Camera* camera; 					/**<Pointer on the scene camera*/
	glm::mat4 modelview_matrix;			/**<Mesh's view x model matrix*/ 
	bool face_culling_enabled ;			/**<Is culling enabled*/
	bool depth_mask_enabled ; 			/**<Is depth enabled*/ 
	bool is_drawn;						/**<Is the mesh */
	Shader* shader_program; 			/**<Shader to be used for the mesh*/
	RASTERMODE polygon_mode;
};

/*****************************************************************************************************************/

/**
 * @brief A generic cube
 * 
 */
class CubeMesh : public Mesh{
public:

	/**
	 * @brief Construct a new Cube  Mesh object
	 * 
	 */
	CubeMesh(SceneNodeInterface* parent = nullptr); 

	/**
	 * @brief 
	 * 
	 */
	void preRenderSetup();
	
	/**
	 * @brief Destroy the Cube  Mesh object
	 * 
	 */
	virtual ~CubeMesh();
};

/*****************************************************************************************************************/


/**
 * @brief Cubemap Mesh class
 * 
 */
class CubeMapMesh : public CubeMesh{
public:
	/**
	 * @brief 
	 * 
	 * @return glm::mat4 
	 */
	glm::mat4 computeFinalTransformation() override ; 

public:

	/**
	 * @brief Construct a new Cube Map Mesh object
	 * 
	 */
	CubeMapMesh(SceneNodeInterface* parent = nullptr); 


	/**
	 * @brief Destroy the Cube Map Mesh object
	 * 
	 */
	virtual ~CubeMapMesh();

	/**
	 * @brief Computes relevant matrices , and sets up the culling + depth states
	 * 
	 */
	virtual void preRenderSetup();
	
};

/*****************************************************************************************************************/
class QuadMesh : public Mesh{
public:
	
	/**
	 * @brief Construct a new Quad Mesh object
	 * 
	 * @param parent 
	 */
	QuadMesh(SceneNodeInterface* parent = nullptr);
	
	/**
	 * @brief Destroy the Quad Mesh object
	 * 
	 */
	virtual ~QuadMesh(); 
	
	/**
	 * @brief 
	 * 
	 */
	void preRenderSetup() override;
	
};


/*****************************************************************************************************************/

/**
 * @class FrameBufferMesh
 * @brief Mesh with an FBO or/and RBO attached to it  
 * 
 */
class FrameBufferMesh : public QuadMesh{
public:
	
	/**
	 * @brief Construct a new FrameBufferMesh 
	 * 
	 */
	FrameBufferMesh(); 

	/**
	 * @brief Construct a new Frame Buffer Mesh 
	 * 
	 * @param database_texture_index The index of the framebuffer texture in the database
	 */
	FrameBufferMesh(int database_texture_index , Shader* shader); 

	/**
	 * @brief Computes relevant matrices , and sets up the culling + depth states
	 * 
	 */
	virtual void preRenderSetup();

	/**
	 * @brief Destroy the Frame Buffer Mesh object
	 * 
	 */
	virtual ~FrameBufferMesh(); 

	
};


/*****************************************************************************************************************/

class BoundingBoxMesh : public Mesh{
public:
	
	/**
	 * @brief Construct a new Bounding Box Mesh object
	 * @param parent Predecessor in the scene graph 
	 */
	BoundingBoxMesh(SceneNodeInterface* parent = nullptr); 

	/**
	 * @brief Construct a new Bounding Box Mesh object
	 * 
	 * @param bound_mesh 
	 * @param display_shader 
	 */
	BoundingBoxMesh(Mesh* bound_mesh , Shader* display_shader);

	/**
	 * @brief Construct a new Bounding Box Mesh using pre-computed bounding boxes
	 * 
	 * @param bound_mesh The Mesh we want to wrap in an aabb . Note that this mesh is used as parent in the scene graph
	 * @param bounding_box The pre-computed bounding box 
	 * @param display_shader Shader used to display the bounding box 
	 */
	BoundingBoxMesh(Mesh* bound_mesh , const BoundingBox& bounding_box , Shader* display_shader); 
	/**
	 * @see Mesh::afterRenderSetup()
	 */
	virtual void afterRenderSetup() override ; 
	/**
	 * 
	 * @see Mesh::preRenderSetup()
	 */
	virtual void preRenderSetup(); 
	
	/**
	 * @brief Destroy the Bounding Box Mesh object
	 * 
	 */
	virtual ~BoundingBoxMesh();

	virtual BoundingBox getBoundingBoxObject(){return bounding_box;}
protected:
	BoundingBox bounding_box;  
};

















#endif
