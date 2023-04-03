#include "../includes/Shader.h"
#include <QMatrix4x4>

inline void shaderErrorCheck(QOpenGLShaderProgram* shader_program){
	if(shader_program){
		QString log = shader_program->log(); 
		std::cout << log.constData() << std::endl; 
	}
}

Shader::Shader(){
	camera_pointer = nullptr ; 
}

Shader::~Shader(){
	
}

void Shader::enableAttributeArray(GLuint att){
	shader_program->enableAttributeArray(att); 
}

void Shader::setAttributeBuffer(GLuint location , GLenum type , int offset , int tuplesize , int stride ){
	shader_program->setAttributeBuffer(location , type , offset , tuplesize , stride); 
}

void Shader::setTextureUniforms(){
	setUniformValue(shader_program->uniformLocation(DiffuseTexture::getTextureTypeCStr()) , Texture::DIFFUSE) ; 
	setUniformValue(shader_program->uniformLocation(NormalTexture::getTextureTypeCStr()) , Texture::NORMAL) ; 
	setUniformValue(shader_program->uniformLocation(MetallicTexture::getTextureTypeCStr()) , Texture::METALLIC) ; 
	setUniformValue(shader_program->uniformLocation(RoughnessTexture::getTextureTypeCStr()) , Texture::ROUGHNESS) ; 
	setUniformValue(shader_program->uniformLocation(AmbiantOcclusionTexture::getTextureTypeCStr()) , Texture::AMBIANTOCCLUSION) ; 	
	setUniformValue(shader_program->uniformLocation(GenericTexture::getTextureTypeCStr()) , Texture::SPECULAR) ; 
	setUniformValue(shader_program->uniformLocation(GenericTexture::getTextureTypeCStr()) , Texture::GENERIC) ; 
	setUniformValue(shader_program->uniformLocation(GenericTexture::getTextureTypeCStr()) , Texture::GENERIC) ; 
}

void Shader::setSceneCameraPointer(Camera* camera){
	camera_pointer = camera;
}

void Shader::updateCamera(){
	if(camera_pointer != nullptr)
		setMatrixUniform(camera_pointer->getViewProjection() , "VP"); 
}


template<class T> 
void Shader::setUniformValue(int location , T value) {
	shader_program->setUniformValue(location , value); 
}

void Shader::setMatrixUniform(glm::mat4 matrix , const char* name){
	QMatrix4x4 q_matrix (glm::value_ptr(matrix)) ; 
	int location = shader_program->uniformLocation(name); 
	assert(location != -1 && "Shader uniform value for matrix is invalid"); 
	setUniformValue(location , q_matrix.transposed()); 
}

void Shader::setMatrixUniform(glm::mat3 matrix , const char* name){
	QMatrix3x3 q_matrix (glm::value_ptr(matrix)); 
	int location = shader_program->uniformLocation(name); 
	assert(location != -1 && "Shader uniform value for matrix is invalid"); 
	setUniformValue(location , q_matrix); 
}

void Shader::initializeShader(){
	shader_program = new QOpenGLShaderProgram(); 
	shader_program->addShaderFromSourceFile(QOpenGLShader::Vertex , "../shaders/simple.vert"); 
	shader_program->addShaderFromSourceFile(QOpenGLShader::Fragment , "../shaders/simple.frag"); 	
	shader_program->link();
	shaderErrorCheck(shader_program) ;
	setTextureUniforms();
}

void Shader::bind(){
	updateCamera(); 
	shader_program->bind(); 
}

void Shader::release(){
	shader_program->release(); 
}

void Shader::clean(){
	if(shader_program != nullptr){
		shader_program->release(); 
		std::cout << "Destroying shader : "<< shader_program  << std::endl ;  
		delete shader_program ;
		shader_program = nullptr ; 
	}
}

