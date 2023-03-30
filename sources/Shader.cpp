#include "../includes/Shader.h"





inline void shaderErrorCheck(QOpenGLShaderProgram* shader_program){
	if(shader_program){
		QString log = shader_program->log(); 
		std::cout << log.constData() << std::endl; 
	}
}

Shader::Shader(){

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
	shader_program->setUniformValue(shader_program->uniformLocation(DiffuseTexture::getTextureTypeCStr()) , Texture::DIFFUSE) ; 
	shader_program->setUniformValue(shader_program->uniformLocation(NormalTexture::getTextureTypeCStr()) , Texture::NORMAL) ; 
	shader_program->setUniformValue(shader_program->uniformLocation(MetallicTexture::getTextureTypeCStr()) , Texture::METALLIC) ; 
	shader_program->setUniformValue(shader_program->uniformLocation(RoughnessTexture::getTextureTypeCStr()) , Texture::ROUGHNESS) ; 
	shader_program->setUniformValue(shader_program->uniformLocation(AmbiantOcclusionTexture::getTextureTypeCStr()) , Texture::AMBIANTOCCLUSION) ; 
	shader_program->setUniformValue(shader_program->uniformLocation(GenericTexture::getTextureTypeCStr()) , Texture::GENERIC) ; 
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
