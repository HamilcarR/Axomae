#include "../includes/GLGeometryBuffer.h"




GLGeometryBuffer::GLGeometryBuffer(){
	geometry = nullptr ; 
	vao = 0 ; 
	vertex_buffer = 0 ; 
	normal_buffer = 0 ; 
	index_buffer = 0 ; 
	texture_buffer = 0 ; 
	color_buffer = 0 ; 
	tangent_buffer = 0 ; 
	buffers_filled = false; 
}

GLGeometryBuffer::GLGeometryBuffer(const Object3D *geometry):GLGeometryBuffer(){
	this->geometry = geometry ; 
}


GLGeometryBuffer::~GLGeometryBuffer(){

}

bool GLGeometryBuffer::isReady() const {
	return vao &&
	    vertex_buffer && 
		normal_buffer && 
		index_buffer && 
		texture_buffer && 
		color_buffer && 
		tangent_buffer && 
		geometry ;  

}

void GLGeometryBuffer::clean(){
	unbindVao(); 
	if (vertex_buffer != 0) glDeleteBuffers(1 , &vertex_buffer); 
	if (normal_buffer != 0) glDeleteBuffers(1 , &normal_buffer); 
	if (index_buffer != 0) glDeleteBuffers(1 , &index_buffer); 
	if (texture_buffer != 0) glDeleteBuffers(1 , &texture_buffer); 
	if (color_buffer != 0) glDeleteBuffers(1 , &color_buffer); 
	if (tangent_buffer != 0) glDeleteBuffers(1 , &tangent_buffer); 
	if (vao != 0) glDeleteVertexArrays(1 , &vao); 
}

void GLGeometryBuffer::initializeBuffers(){
	glGenVertexArrays(1 , &vao);
	bindVao();
	glGenBuffers(1 , &vertex_buffer); 
	glGenBuffers(1 , &normal_buffer); 
	glGenBuffers(1 , &color_buffer); 
	glGenBuffers(1 , &texture_buffer); 
	glGenBuffers(1 , &tangent_buffer); 
	glGenBuffers(1 , &index_buffer); 
}


void GLGeometryBuffer::bind(){
	bindVao();
}

void GLGeometryBuffer::unbind(){
	unbindVao(); 
}

void GLGeometryBuffer::bindVao(){
	glBindVertexArray(vao) ; 
}

void GLGeometryBuffer::bindVertexBuffer(){ 
	glBindBuffer(GL_ARRAY_BUFFER , vertex_buffer); 
} 

void GLGeometryBuffer::bindNormalBuffer(){
	glBindBuffer(GL_ARRAY_BUFFER , normal_buffer); 
} 
void GLGeometryBuffer::bindTextureBuffer(){ 
	glBindBuffer(GL_ARRAY_BUFFER , texture_buffer); 
} 
void GLGeometryBuffer::bindColorBuffer(){ 
	glBindBuffer(GL_ARRAY_BUFFER , color_buffer); 
} 
void GLGeometryBuffer::bindIndexBuffer(){ 
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer); 
}

void GLGeometryBuffer::bindTangentBuffer(){
	glBindBuffer(GL_ARRAY_BUFFER , tangent_buffer); 
}

void GLGeometryBuffer::unbindVao(){
	glBindVertexArray(0) ; 
}

//TODO: [AX-20] Provide methods to fill individual buffer , or to modify them
void GLGeometryBuffer::fillBuffers(){
	if(!buffers_filled){
    	bindVao(); 
    	bindVertexBuffer(); 
    	glBufferData(GL_ARRAY_BUFFER , geometry->vertices.size() * sizeof(float) , geometry->vertices.data() , GL_STATIC_DRAW) ;  
    	bindColorBuffer(); 
    	glBufferData(GL_ARRAY_BUFFER , geometry->colors.size() * sizeof(float) , geometry->colors.data() , GL_STATIC_DRAW) ;  
    	bindNormalBuffer(); 
    	glBufferData(GL_ARRAY_BUFFER , geometry->normals.size() * sizeof(float) , geometry->normals.data() , GL_STATIC_DRAW) ;  
    	bindTextureBuffer() ; 
    	glBufferData(GL_ARRAY_BUFFER , geometry->uv.size() * sizeof(float) , geometry->uv.data() , GL_STATIC_DRAW) ;  
    	bindTangentBuffer(); 
    	glBufferData(GL_ARRAY_BUFFER , geometry->tangents.size() * sizeof(float) , geometry->tangents.data() , GL_STATIC_DRAW) ;  
    	bindIndexBuffer(); 
    	glBufferData(GL_ELEMENT_ARRAY_BUFFER , geometry->indices.size() * sizeof(unsigned int) , geometry->indices.data() , GL_STATIC_DRAW); 
    	errorCheck(__FILE__ , __LINE__); 
		buffers_filled = true; 
	}
}


