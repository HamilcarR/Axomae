#include "../includes/Mesh.h"
#include "../includes/PerformanceLogger.h"


Mesh::Mesh(SceneNodeInterface* parent):SceneTreeNode(parent){
	mesh_initialized = false;
	shader_program = nullptr; 
	name = "uninitialized mesh" ; 
	is_drawn = true; 
}

Mesh::Mesh(const Mesh& copy) : SceneTreeNode(copy){
	geometry = copy.geometry ; 
	material = copy.material ; 
	name = copy.name ; 
	shader_program = copy.shader_program; 
}

Mesh::Mesh(const Object3D& geo , const Material& mat , SceneNodeInterface* parent ) : Mesh(parent){
	geometry = geo; 
	material = mat;
	name = "uninitialized mesh"  ;
	shader_program = nullptr ; 
}

Mesh::Mesh(const std::string& n , const Object3D& geo , const Material& mat  , SceneNodeInterface* parent ) : Mesh(parent){
	geometry = geo; 
	material = mat;
	name = n ; 
	shader_program = nullptr ; 
}

Mesh::Mesh(const std::string& n , const Object3D& geo , const Material& mat , Shader* shader, SceneNodeInterface* parent ) : Mesh(parent){
	geometry = geo ;
	material = mat ; 
	name = n ; 
	shader_program = shader; 
	material.setShaderPointer(shader); 
}

Mesh::Mesh(const std::string& n , const Object3D&& geo , const Material& mat , Shader* shader, SceneNodeInterface* parent ) : Mesh(parent){  
	geometry = std::move(geo); 
	material = mat ; 
	name = n ; 
	shader_program = shader; 
	material.setShaderPointer(shader);
}

Mesh::~Mesh(){}

void Mesh::initializeGlData(){
	if(shader_program != nullptr){
		shader_program->initializeShader();
	shader_program->bind(); 
	material.initializeMaterial();
	shader_program->release(); 
	mesh_initialized = true ; 
	}
}

void Mesh::bindMaterials(){
	material.bind(); 
}

void Mesh::unbindMaterials(){
	material.unbind(); 
}

void Mesh::preRenderSetup(){
	setFaceCulling(true); 
	cullBackFace();
	face_culling_enabled = true ; 
	setDepthMask(true); 
	setDepthFunc(LESS); 
	depth_mask_enabled = true ;
	setParent(camera) ; 
	glm::mat4 model_mat = getWorldSpaceModelMatrix();
	modelview_matrix = camera->getView() * model_mat ;
	if(shader_program){	
		shader_program->setSceneCameraPointer(camera); 
		shader_program->setAllMatricesUniforms(model_mat) ; 	
	}
}

void Mesh::setupAndBind(){
	if(shader_program){
		bindShaders();
		errorCheck(__FILE__ , __LINE__); 
		preRenderSetup(); 
		errorCheck(__FILE__ , __LINE__); 
	}
}

void Mesh::bindShaders(){
	if(shader_program)
		shader_program->bind(); 	
}

void Mesh::releaseShaders(){
	if(shader_program)
		shader_program->release(); 
}
void Mesh::clean(){
	shader_program = nullptr; 
	material.clean();
}

bool Mesh::isInitialized(){
	return mesh_initialized; 	
}

void Mesh::setSceneCameraPointer(Camera *camera){
	this->camera = camera ;	
}

void Mesh::cullBackFace(){
	glCullFace(GL_BACK); 
}

void Mesh::cullFrontFace(){
	glCullFace(GL_FRONT); 
}

void Mesh::cullFrontAndBackFace(){
	glCullFace(GL_FRONT_AND_BACK); 
}	

void Mesh::afterRenderSetup(){
	return ;
}

void Mesh::setPolygonDrawMode(RASTERMODE mode){
	glPolygonMode(GL_FRONT_AND_BACK , mode); 
}

void Mesh::setFaceCulling(bool value){
	if(value){
		glEnable(GL_CULL_FACE); 
		face_culling_enabled = true; 
	}
	else{
		glDisable(GL_CULL_FACE);
		face_culling_enabled = false; 
	}
}

void Mesh::setDepthMask(bool val){
	glDepthMask(val ? GL_TRUE : GL_FALSE) ;
	depth_mask_enabled = val ; 
}

void Mesh::setDepthFunc(DEPTHFUNC func){
	glDepthFunc(func) ; 
}

/*****************************************************************************************************************/

CubeMapMesh::CubeMapMesh(SceneNodeInterface* parent) : Mesh(parent) {
	std::vector<float> vertices = { 
		-1 , -1 , -1 	,  // 0
		1 , -1 , -1 	,  // 1
		-1 , 1 , -1 	,   // 2
		1 , 1 , -1 		,   // 3
		-1 , -1 , 1 	,   // 4
		1 , -1 , 1 		,   // 5
		-1 , 1 , 1 		,    // 6
		1 , 1 , 1 		};   // 7

	std::vector<unsigned int> indices = { 
		0 , 1 , 2 	, //Front face
		1 , 3 , 2 	, //
		5 , 4 , 6 	, //Back face
		6 , 7 , 5 	, //
		0 , 2 , 6 	, //Left face
		0 , 6 , 4	, //
		1 , 5 , 7	, //Right face
		7 , 3 , 1	, //
		3 , 7 , 6 	, //Up face
		2 , 3 , 6 	, //
		0 , 4 , 5 	, //Down face
		0 , 5 , 1   };

	std::vector<float> textures = { 
		0 , 0 		, 
		1 , 0 		, 
		0 , 1 		, 
		1 , 1 		, 
		0 , 0 		, 
		1 , 0 		, 
		0 , 1 		, 
		1 , 1 		};

	std::vector<float> colors = { 
		1 , 0 , 0 	,
		1 , 0 , 0 	, 	
		1 , 0 , 0 	, 
		1 , 0 , 0 	,
		1 , 0 , 0 	, 	
		1 , 0 , 0 	, 
		1 , 0 , 0 	, 
		1 , 0 , 0 	};
	
	geometry.indices = indices ; 
	geometry.vertices = vertices ;
	geometry.uv = textures ; 
	geometry.colors = colors ; 
	local_modelmatrix = glm::mat4(1.f);
	name="CubeMap" ;
}

CubeMapMesh::~CubeMapMesh(){
}

void CubeMapMesh::preRenderSetup(){
	setFaceCulling(false);
	setDepthFunc(LESS_OR_EQUAL); 	
	glm::mat4 view = glm::mat4(glm::mat3(camera->getView()));
	glm::mat4 projection = camera->getProjection() ; 
	local_modelmatrix = camera->getSceneRotationMatrix() ;  				
	if(shader_program != nullptr){
		shader_program->setSceneCameraPointer(camera); 	
		shader_program->setAllMatricesUniforms(projection , view , local_modelmatrix) ; 
	}
}

/*****************************************************************************************************************/

FrameBufferMesh::FrameBufferMesh():Mesh(){

	std::vector<float> vertices = {  
		-1.0f, -1.0f, 0.f, 
		-1.0f, 1.0f, 0.f ,  
		1.0f, 1.0f, 0.f ,  
		1.0f, -1.0f, 0.f 
	};
	std::vector<unsigned int> indices = {
		2 , 1 , 0 , 
		3 , 2 , 0
	};
	std::vector<float> textures = {
		0 , 0 , 
		0 , 1 , 
		1 , 1 ,
		1 , 0  
	};
	geometry.indices = indices; 
	geometry.vertices = vertices ; 
	geometry.uv = textures ;
	name = "Custom screen framebuffer";
	local_modelmatrix = glm::mat4(1.f);   
}

FrameBufferMesh::FrameBufferMesh(int texture_index , Shader* _shader): FrameBufferMesh(){	
	shader_program = _shader; 	 
	material.setShaderPointer(shader_program);  
	material.addTexture(texture_index , Texture::FRAMEBUFFER);	
}

FrameBufferMesh::~FrameBufferMesh(){

}

void FrameBufferMesh::preRenderSetup(){
	setFaceCulling(false);
	face_culling_enabled = false;
	setDepthFunc(ALWAYS); 
}


/*****************************************************************************************************************/

BoundingBoxMesh::BoundingBoxMesh(SceneNodeInterface* parent):Mesh(parent){

}

//TODO: [AX-19] Fix cubemap bounding box computation 
BoundingBoxMesh::BoundingBoxMesh(Mesh* m , Shader* s) : BoundingBoxMesh(m){
	shader_program = s ;
	name = std::string("Boundingbox-") + m->getMeshName(); 
	std::vector<float> vertices = m->getGeometry().vertices; 
	bounding_box = BoundingBox(vertices);
	material.setShaderPointer(s);
	std::pair<std::vector<float> , std::vector<unsigned>> geom = bounding_box.getVertexArray(); 
	geometry.vertices = geom.first; 
	geometry.indices = geom.second;  
}

BoundingBoxMesh::BoundingBoxMesh(Mesh* m , const BoundingBox& bbox , Shader* s) : BoundingBoxMesh(m){
	shader_program = s ; 
	name = std::string("Boundingbox-") + m->getMeshName(); 
	bounding_box = bbox ;
	material.setShaderPointer(s); 
	std::pair<std::vector<float> , std::vector<unsigned>> geom = bounding_box.getVertexArray(); 
	geometry.vertices = geom.first ; 
	geometry.indices = geom.second ;  
}

BoundingBoxMesh::~BoundingBoxMesh(){

}

void BoundingBoxMesh::preRenderSetup(){
	setFaceCulling(true); 
	face_culling_enabled = true;
	setDepthMask(true); 
	setDepthFunc(LESS);
	setPolygonDrawMode(LINE); 
	depth_mask_enabled = true;
	glm::mat4 model = getWorldSpaceModelMatrix(); 
	modelview_matrix = camera->getView() * model ; 
	bounding_box = modelview_matrix * bounding_box ;  
	if(shader_program){	
		shader_program->setSceneCameraPointer(camera); 
		shader_program->setAllMatricesUniforms(model) ; 	
	}
}


void BoundingBoxMesh::afterRenderSetup(){
	setPolygonDrawMode(FILL); 
}



































