#include "../includes/Shader.h"
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
	camera_pointer = nullptr ; 
}

Shader::Shader(const std::string vertex_code , const std::string fragment_code){
	fragment_shader_txt = fragment_code ; 
	vertex_shader_txt = vertex_code ; 
	camera_pointer = nullptr ; 
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
	setUniform(GenericTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::SPECULAR)) ; 
	setUniform(GenericTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::GENERIC)) ; 
	setUniform(GenericTexture::getTextureTypeCStr() ,  static_cast<int> (Texture::GENERIC)) ; 

}

void Shader::setSceneCameraPointer(Camera* camera){
	camera_pointer = camera;
}

void Shader::updateCamera(){
	if(camera_pointer != nullptr){
		glUseProgram(shader_program) ; 
		setMatrixUniform("VP" , camera_pointer->getViewProjection()); 
	}
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

void Shader::setMatrixUniform(const char* name , const glm::mat4 &matrix){
	int location =  glGetUniformLocation(shader_program , name); 
	glUniformMatrix4fv(location ,1 , GL_FALSE , glm::value_ptr(matrix)); 
}

void Shader::setMatrixUniform(const char* name , const glm::mat3 &matrix){
	int location = glGetUniformLocation(shader_program , name); 
	glUniformMatrix3fv(location ,1 , GL_FALSE , glm::value_ptr(matrix)); 
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
}

void Shader::bind(){
	updateCamera(); 
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

