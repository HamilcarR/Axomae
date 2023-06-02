#include "../includes/Shader.h"
#include "../includes/UniformNames.h"
#include <QMatrix4x4>
#include <cstring>

#define SHADER_ERROR_LOG_SIZE 512


/**
 * @file Shader.cpp
 * Implements functions and methods relative to the shading 
 * 
 */


static int success  ; 
static char infoLog[SHADER_ERROR_LOG_SIZE] ; 


/**
 * This function checks for compilation errors in a shader and prints an error message if there are
 * any.
 * 
 * @param shader_id The ID of the shader object that needs to be checked for compilation errors.
 */
inline void shaderCompilationErrorCheck(unsigned int shader_id){
	success = 0; 
	memset(infoLog , 0 , SHADER_ERROR_LOG_SIZE) ; 
	glGetShaderiv(shader_id , GL_COMPILE_STATUS , &success) ; 
	if(!success){
		glGetShaderInfoLog(shader_id , SHADER_ERROR_LOG_SIZE , nullptr , infoLog);
		std::cerr << "Shader compilation failed with error : " << infoLog << "\n" ; 
	}
}

/**
 * This function checks for errors in shader program linking and prints an error message if there is a
 * failure.
 * 
 * @param program_id The ID of the shader program that needs to be checked for linking errors.
 */
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

void Shader::setTextureUniforms(std::string texture_name , Texture::TYPE type){
	glUseProgram(shader_program) ; 
	setUniform(texture_name , static_cast<int>(type)); 
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
		setUniform(uniform_name_vec3_camera_position , camera_pointer->getPosition()); 	
}

void Shader::setAllMatricesUniforms(const glm::mat4& model){
	setModelViewProjection(model); 
	setNormalMatrixUniform(model);
	setInverseModelMatrixUniform(model);
	if(camera_pointer != nullptr) 
		setInverseModelViewMatrixUniform(camera_pointer->getView() , model);  
}

void Shader::setAllMatricesUniforms(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model){
	setModelViewProjection(projection , view , model); 
	setNormalMatrixUniform(model); 
}

void Shader::setInverseModelViewMatrixUniform(const glm::mat4& view , const glm::mat4& model){
	glm::mat4 inverse = glm::inverse(view * model); 
	setUniform(uniform_name_matrix_inverse_modelview , inverse); 
}

void Shader::setNormalMatrixUniform(const glm::mat4& model){
	setUniform(uniform_name_matrix_normal , glm::mat3(glm::transpose(glm::inverse(camera_pointer->getView() * model)))); 
}

void Shader::setInverseModelMatrixUniform(const glm::mat4& model){
	auto inverse_model = glm::inverse(model) ; 
	setUniform(uniform_name_matrix_inverse_model , inverse_model); 
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

void Shader::setUniformValue(int location , const int value) {
	glUniform1i(location , value); 
}

void Shader::setUniformValue(int location , const float value){
	glUniform1f(location , value); 
}

void Shader::setUniformValue(int location , const unsigned int value){
	glUniform1ui(location , value); 
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
	errorCheck(); 
}
void Shader::recompile(){
	clean(); 
	initializeShader(); 
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

/***********************************************************************************************************************************************************/

ScreenFrameBufferShader::ScreenFrameBufferShader() : Shader() {
	type = SCREEN_FRAMEBUFFER ; 
}

ScreenFrameBufferShader::ScreenFrameBufferShader(const std::string vertex , const std::string frag) : Shader(vertex , frag) {
	type = SCREEN_FRAMEBUFFER ; 
}

ScreenFrameBufferShader::~ScreenFrameBufferShader(){

}
/*
void ScreenFrameBufferShader::setTextureUniforms(){
	bind(); 
	setUniform(FrameBufferTexture::getTextureTypeCStr() , static_cast<int>(Texture::FRAMEBUFFER)); 
}
*/