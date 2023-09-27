#include "../includes/Mesh.h"
#include "../includes/PerformanceLogger.h"


Mesh::Mesh(SceneNodeInterface* parent):SceneTreeNode(parent){
	mesh_initialized = false;
	face_culling_enabled = false;
	depth_mask_enabled = false; 
	is_drawn = false ; 
	polygon_mode = FILL ; 
	shader_program = nullptr; 
	name = "uninitialized mesh" ; 
	is_drawn = true;
	polygon_mode = FILL ; 
	modelview_matrix = glm::mat4(1.f);  
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
	glPolygonMode(GL_FRONT_AND_BACK , polygon_mode); 
	depth_mask_enabled = true ;
	glm::mat4 model_mat = computeFinalTransformation();  
	modelview_matrix = camera->getView() * model_mat ;
	if(shader_program){	
		shader_program->setSceneCameraPointer(camera); 
		shader_program->setAllMatricesUniforms(model_mat) ; 
		if(cubemap_reference)
			shader_program->setCubemapNormalMatrixUniform(camera->getView() * cubemap_reference->getAccumulatedModelMatrix()); 	
	}
}

void Mesh::setShader(Shader* shader){
	assert(shader != nullptr); 
	shader_program = shader;
	if(!shader_program->isInitialized())
		shader_program->initializeShader(); 
	material.setShaderPointer(shader); 

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
	if(shader_program)
		shader_program->setSceneCameraPointer(camera); 	
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
	polygon_mode = mode; 
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
CubeMesh::CubeMesh(SceneNodeInterface* parent) : Mesh(parent) {
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
	local_transformation = glm::mat4(1.f);
	name="Generic-Cube" ;
}

CubeMesh::~CubeMesh(){
}

void CubeMesh::preRenderSetup(){
	setFaceCulling(true); 
	setDepthMask(true); 
	setDepthFunc(ALWAYS);
	glPolygonMode(GL_FRONT_AND_BACK , FILL); 
	glm::mat4 model_mat = computeFinalTransformation(); 
	modelview_matrix = camera->getView() * model_mat ;
	if(shader_program){	
		shader_program->setSceneCameraPointer(camera); 
		shader_program->setAllMatricesUniforms(camera->getProjection() , camera->getView() , model_mat) ; 	
	}
}



/*****************************************************************************************************************/
CubeMapMesh::CubeMapMesh(SceneNodeInterface* parent) : CubeMesh(parent) {
		name="CubeMap" ;
}

CubeMapMesh::~CubeMapMesh(){
}

void CubeMapMesh::preRenderSetup(){
	setFaceCulling(false);
	setDepthFunc(LESS_OR_EQUAL); 
	glPolygonMode(GL_FRONT_AND_BACK , FILL); 	
	glm::mat4 view = glm::mat4(glm::mat3(camera->getView()));
	glm::mat4 projection = camera->getProjection() ; 
	glm::mat4 local = camera->getSceneRotationMatrix() ; 
	local = glm::scale(local , glm::vec3(INT_MAX , INT_MAX , INT_MAX)); 				
	if(shader_program != nullptr){
		shader_program->setSceneCameraPointer(camera); 	
		shader_program->setAllMatricesUniforms(projection , view , local) ; 
	}
}

glm::mat4 CubeMapMesh::computeFinalTransformation(){
	accumulated_transformation = getParent()->computeFinalTransformation(); 
	return accumulated_transformation; 
}
/*****************************************************************************************************************/
QuadMesh::QuadMesh(SceneNodeInterface* parent): Mesh(parent){
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
	local_transformation = glm::mat4(1.f);
	name = "Quad"; 
}

QuadMesh::~QuadMesh(){

}

void QuadMesh::preRenderSetup(){
	setFaceCulling(false); 
	setDepthMask(true); 
	setDepthFunc(LESS);
	glPolygonMode(GL_FRONT_AND_BACK , FILL); 
	glm::mat4 model_mat = computeFinalTransformation(); 
	modelview_matrix = camera->getView() * model_mat ;
	if(shader_program){	
		shader_program->setSceneCameraPointer(camera); 
		shader_program->setAllMatricesUniforms(camera->getProjection() , camera->getView() , model_mat) ; 	
	}
}





/*****************************************************************************************************************/

FrameBufferMesh::FrameBufferMesh():QuadMesh(){
	name = "Custom screen framebuffer";
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
	glPolygonMode(GL_FRONT_AND_BACK , FILL); 
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
	glPolygonMode(GL_FRONT_AND_BACK , LINE); 
	depth_mask_enabled = true;
	glm::mat4 model = computeFinalTransformation(); 
	modelview_matrix = camera->getView() * model ; 
	bounding_box = modelview_matrix * bounding_box ;  
	if(shader_program){	
		shader_program->setSceneCameraPointer(camera); 
		shader_program->setAllMatricesUniforms(model) ; 	
	}
}


void BoundingBoxMesh::afterRenderSetup(){
	glPolygonMode(GL_FRONT_AND_BACK , polygon_mode); 
}



































