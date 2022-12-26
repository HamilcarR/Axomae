#include "../includes/Drawable.h"


using namespace axomae ;


Drawable::Drawable(){
	mesh_object = nullptr; 
}

Drawable::Drawable(Mesh *mesh){
	Mesh m = *mesh ; 
	mesh_object = new Mesh(m); 
	initialize() ; 
}

Drawable::~Drawable(){

}

void Drawable::clean(){
	if(shader_program != nullptr){
		std::cout << "Destroying shader : "<< shader_program  << std::endl ;  
		delete shader_program ;
		shader_program = nullptr ; 
	}
	if(vao.isCreated()){
		vao.release(); 
		vao.destroy(); 
	}
	vertex_buffer.destroy() ; 
	normal_buffer.destroy() ; 
	index_buffer.destroy() ; 
	texture_buffer.destroy() ;
	
	glDeleteTextures(1 , &sampler2D) ; 
}


bool Drawable::ready(){
	bool vao_created = vao.isCreated() ; 
	bool buffers_created = vertex_buffer.isCreated() && 
			       normal_buffer.isCreated() && 
			       index_buffer.isCreated() && 
			       texture_buffer.isCreated() ; //TODO : add color buffer if useful ? 
	bool texture_created = sampler2D != 0 ; 
	return vao_created && buffers_created && texture_created ; 
}

bool Drawable::initialize(){
	if(mesh_object == nullptr)
		return false ; 
	shader_program = new QOpenGLShaderProgram(); 
	shader_program->addShaderFromSourceFile(QOpenGLShader::Vertex , "../shaders/simple.vert"); 
	shader_program->addShaderFromSourceFile(QOpenGLShader::Fragment , "../shaders/simple.frag"); 	
	shader_program->link();
	shaderErrorCheck(shader_program) ; 
	bool vao_okay = vao.create();
	assert(vao_okay == true) ; 
	vertex_buffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer); 	
	texture_buffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer); 
	index_buffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
	normal_buffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	color_buffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer); 
	vertex_buffer.create();
	color_buffer.create();
	texture_buffer.create(); 
	index_buffer.create();
	normal_buffer.create();
	vertex_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw);	
	texture_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw); 
	index_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	normal_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw); 
	color_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw); 
	vao.bind(); 
	vertex_buffer.bind(); 
	index_buffer.bind();
	shader_program->bind(); 
	shader_program->enableAttributeArray(0);
	shader_program->setAttributeBuffer(0 , GL_FLOAT , 0 , 3 , 0 ) ;
	texture_buffer.bind(); 
	shader_program->enableAttributeArray(1) ; 
	shader_program->setAttributeBuffer(1 , GL_FLOAT , 0 , 2 , 0 ) ; 
	vao.release(); 
	shader_program->release();
	vertex_buffer.release();
	index_buffer.release();
	
	/*manage in texture class*/
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	glTexImage2D(GL_TEXTURE_2D , 0 , GL_RGBA , mesh_object->material.textures.diffuse.width , mesh_object->material.textures.diffuse.height , 0 , GL_BGRA , GL_UNSIGNED_BYTE , mesh_object->material.textures.diffuse.data); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	shader_program->setUniformValue(shader_program->uniformLocation("diffuse") , 0) ; 
	return true ; 
}

void Drawable::start_draw(){
	if(mesh_object != nullptr){
		vertex_buffer.bind();
		vertex_buffer.allocate(mesh_object->geometry.vertices.data() , mesh_object->geometry.vertices.size() * sizeof(float)); 
		vertex_buffer.release();
		texture_buffer.bind();
		texture_buffer.allocate(mesh_object->geometry.uv.data() , mesh_object->geometry.uv.size() * sizeof(float)) ; 
		texture_buffer.release(); 
		index_buffer.bind();
		index_buffer.allocate(mesh_object->geometry.indices.data() , mesh_object->geometry.indices.size() * sizeof(unsigned int)); 
		index_buffer.release();
		shader_program->bind();
		vao.bind();
	}

}

void Drawable::end_draw(){
	vao.release();
	shader_program->release();


}

void Drawable::bind(){
	shader_program->bind(); 
	vao.bind();
	glBindTexture(GL_TEXTURE_2D , sampler2D); 


}

void Drawable::unbind(){
	vao.release();
	shader_program->release();
}


