#include "../includes/Shader.h"
#include "../includes/UniformNames.h"
#include <QMatrix4x4>
#include <cstring>

#define SHADER_ERROR_LOG_SIZE 512

static int success  ; 
static char infoLog[SHADER_ERROR_LOG_SIZE] ; 


inline void shaderCompilationErrorCheck(unsigned int shader_id){
	success = 0; 
	memset(infoLog , 0 , SHADER_ERROR_LOG_SIZE) ; 
	glGetShaderiv(shader_id , GL_COMPILE_STATUS , &success) ; 
	if(!success){
		glGetShaderInfoLog(shader_id , SHADER_ERROR_LOG_SIZE , nullptr , infoLog);
		std::cerr << "Shader compilation failed with error : " << infoLog << "\n" ; 
	}
}

inline void programLinkingErrorCheck(unsigned int program_id){
	success = 0; 
	memset(infoLog , 0 , SHADER_ERROR_LOG_SIZE) ; 
	glGetProgramiv(program_id, GL_LINK_STATUS , &success) ; 
	if(!success){
		glGetProgramInfoLog(program_id , SHADER_ERROR_LOG_SIZE , nullptr , infoLog); 
		std::cerr << "Shader linkage failed with error : " << infoLog << "\n" ; 
	}
}

Shader::Shader(){
	type = GENERIC ; 
}

Shader::Shader(const std::string vertex_code , const std::string fragment_code){
	type = GENERIC; 
	fragment_shader_txt = fragment_code ; 
	vertex_shader_txt = vertex_code ; 
}

Shader::~Shader(){	
}

void Shader::enableAttributeArray(GLuint att){
	glEnableVertexAttribArray(att); 
}

void Shader::setAttributeBuffer(GLuint location , GLenum type , int offset , int tuplesize , int stride ){
	glVertexAttribPointer(location , tuplesize , type , GL_FALSE , stride , (void*) 0); 
}

void Shader::setTextureUniforms(){
	glUseProgram(shader_program) ; 
	setUniform(DiffuseTexture::getTextureTypeCStr() , static_cast<int> (Texture::DIFFUSE)) ; 
	setUniform(NormalTexture::getTextureTypeCStr() , static_cast<int> (Texture::NORMAL)) ; 
	setUniform(MetallicTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::METALLIC)) ; 
	setUniform(RoughnessTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::ROUGHNESS)) ; 
	setUniform(AmbiantOcclusionTexture::getTextureTypeCStr() , static_cast<int> (Texture::AMBIANTOCCLUSION)) ; 	
	setUniform(SpecularTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::SPECULAR)) ; 
	setUniform(EmissiveTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::EMISSIVE)) ; 
	setUniform(CubeMapTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::CUBEMAP)) ; 
	setUniform(GenericTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::GENERIC)) ; 
}

void Shader::setSceneCameraPointer(Camera* camera){
	camera_pointer = camera;
}

void Shader::updateCamera(){
	if(camera_pointer != nullptr)
		setCameraPositionUniform(); 
}

void Shader::setCameraPositionUniform(){
	if(camera_pointer != nullptr)
		setUniform(uniform_name_vector_camera_position , camera_pointer->getPosition()); 
	
}

void Shader::setAllMatricesUniforms(const glm::mat4& model){
	setModelViewProjection(model); 
	setNormalMatrixUniform(model); 
}

void Shader::setAllMatricesUniforms(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model){
	setModelViewProjection(projection , view , model); 
	setNormalMatrixUniform(model); 
}

void Shader::setNormalMatrixUniform(const glm::mat4& model){
	setUniform(uniform_name_matrix_normal , glm::mat3(glm::transpose(glm::inverse(camera_pointer->getView() * model)))); 
}

void Shader::setModelMatrixUniform(const glm::mat4& matrix){
	setUniform(uniform_name_matrix_model , matrix) ; 
}

void Shader::setModelViewProjectionMatricesUniforms(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model){
		glm::mat4 mvp = projection * view * model ;	
		glm::mat4 modelview_matrix = view * model ; 
		glm::mat4 view_projection = projection * view ; 
		setUniform(uniform_name_matrix_modelview , modelview_matrix); 
		setUniform(uniform_name_matrix_model_view_projection , mvp) ; 
		setUniform(uniform_name_matrix_view_projection , view_projection);
		setUniform(uniform_name_matrix_model , model); 
		setUniform(uniform_name_matrix_view , view); 
		setUniform(uniform_name_matrix_projection , projection); 

}

void Shader::setModelViewProjection(const glm::mat4& model){
	if(camera_pointer != nullptr){
		glm::mat4 view = camera_pointer->getView(); 	
		glm::mat4 projection = camera_pointer->getProjection(); 
		updateCamera(); 
		setModelViewProjectionMatricesUniforms(projection , view , model); 
	}
}

void Shader::setModelViewProjection(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model) {
		updateCamera();
		setModelViewProjectionMatricesUniforms(projection , view , model); 
}


template<typename T> 
void Shader::setUniform(const char* name , const T value){
	glUseProgram(shader_program) ; 
	int location = glGetUniformLocation(shader_program , name);
	setUniformValue(location , value);
}

void Shader::setUniformValue(int location , const int value) {
	glUniform1i(location , value); 
}

void Shader::setUniformValue(int location , const glm::mat4 &matrix){
	glUniformMatrix4fv(location ,1 , GL_FALSE , glm::value_ptr(matrix)); 
}

void Shader::setUniformValue(int location , const glm::mat3 &matrix){
	glUniformMatrix3fv(location ,1 , GL_FALSE , glm::value_ptr(matrix)); 
}

void Shader::setUniformValue(int location , const glm::vec4& value){
	glUniform4f(location , value.x , value.y , value.z , value.w); 
}
void Shader::setUniformValue(int location , const glm::vec3& value){
	glUniform3f(location , value.x , value.y , value.z); 
}
void Shader::setUniformValue(int location , const glm::vec2& value){	
	glUniform2f(location , value.x , value.y); 
}

void Shader::initializeShader(){
	const char* vertex_shader_source = (const char*) vertex_shader_txt.c_str() ; 
	const char* fragment_shader_source = (const char*) fragment_shader_txt.c_str(); 
	vertex_shader = glCreateShader(GL_VERTEX_SHADER) ; 
	glShaderSource(vertex_shader , 1 , &vertex_shader_source , nullptr) ; 
	glCompileShader(vertex_shader); 
	shaderCompilationErrorCheck(vertex_shader) ; 
	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER) ; 
	glShaderSource(fragment_shader , 1 , &fragment_shader_source , nullptr) ; 
	glCompileShader(fragment_shader); 
	shaderCompilationErrorCheck(fragment_shader) ; 
	shader_program = glCreateProgram();
	glAttachShader(shader_program , vertex_shader); 
	glAttachShader(shader_program , fragment_shader);
	glLinkProgram(shader_program) ; 
	programLinkingErrorCheck(shader_program) ; 
	setTextureUniforms();
	errorCheck(); 
}

void Shader::bind(){
	glUseProgram(shader_program);  
}

void Shader::release(){
	glUseProgram(0); 
}

void Shader::clean(){
	if(shader_program != 0){	
		std::cout << "Destroying shader : "<< shader_program  << std::endl ;  
		glDeleteShader(vertex_shader) ; 
		glDeleteShader(fragment_shader) ; 
		glDeleteProgram(shader_program) ; 
		shader_program = 0 ; 
	}
}

/***********************************************************************************************************************************************************/

BlinnPhongShader::BlinnPhongShader():Shader(){
	type = BLINN ; 	
}


BlinnPhongShader::BlinnPhongShader(const std::string vertex_code, const std::string fragment_code) : Shader(vertex_code , fragment_code) {
	type = BLINN ; 
}

BlinnPhongShader::~BlinnPhongShader(){

}

/***********************************************************************************************************************************************************/

CubeMapShader::CubeMapShader() : Shader() {
	type = CUBEMAP ; 
}

CubeMapShader::CubeMapShader(const std::string vertex , const std::string frag) : Shader(vertex , frag) {
	type = CUBEMAP ; 
}

CubeMapShader::~CubeMapShader(){

}
