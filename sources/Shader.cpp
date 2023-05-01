#include "../includes/Shader.h"
#include <QMatrix4x4>
#include <cstring>

#define SHADER_ERROR_LOG_SIZE 512

static int success  ; 
static char infoLog[SHADER_ERROR_LOG_SIZE] ; 

/*Shader data names*/
constexpr const char uniform_name_matrix_model[] = "model"; 
constexpr const char uniform_name_matrix_view[] = "view" ; 
constexpr const char uniform_name_matrix_projection[] = "projection"; 
constexpr const char uniform_name_matrix_view_projection[] = "VP"; 
constexpr const char uniform_name_matrix_model_view_projection[] = "MVP" ; 

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

}

Shader::Shader(const std::string vertex_code , const std::string fragment_code){
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
	if(camera_pointer != nullptr){
		setMatrixUniform(uniform_name_matrix_view_projection , camera_pointer->getViewProjection()); 
		setMatrixUniform(uniform_name_matrix_view , camera_pointer->getView()); 
	}
}

void Shader::setModelMatrixUniform(const glm::mat4& matrix){
	setMatrixUniform(uniform_name_matrix_model , matrix) ; 
}

void Shader::setModelViewProjection(const glm::mat4& model){
	if(camera_pointer != nullptr){
		glm::mat4 view_projection = camera_pointer->getViewProjection(); 
		glm::mat4 view_matrix = camera_pointer->getView(); 
		glm::mat4 mvp = view_projection * model ;
		setMatrixUniform(uniform_name_matrix_view , view_matrix); 
		setMatrixUniform(uniform_name_matrix_model_view_projection , mvp) ; 
		setMatrixUniform(uniform_name_matrix_view_projection , view_projection); 
	}
}


void Shader::setModelViewProjection(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model) {
		glm::mat4 view_projection = projection * view ; 
		glm::mat4 mvp = view_projection * model ;
		setMatrixUniform(uniform_name_matrix_model , model); 		
		setMatrixUniform(uniform_name_matrix_view , view); 
		setMatrixUniform(uniform_name_matrix_model_view_projection , mvp) ; 
		setMatrixUniform(uniform_name_matrix_view_projection , view_projection); 
		setMatrixUniform(uniform_name_matrix_projection , projection); 
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

