#include "../includes/GLBuffers.h"




GLBuffers::GLBuffers(){
	geometry = nullptr ; 
	vao = 0 ; 
	vertex_buffer = 0 ; 
	normal_buffer = 0 ; 
	index_buffer = 0 ; 
	texture_buffer = 0 ; 
	color_buffer = 0 ; 
	tangent_buffer = 0 ; 

}

GLBuffers::GLBuffers(const axomae::Object3D *geometry){
	this->geometry = geometry ; 
	vao = 0 ; 
	vertex_buffer = 0 ; 
	normal_buffer = 0 ; 
	index_buffer = 0 ; 
	texture_buffer = 0 ; 
	color_buffer = 0 ; 
	tangent_buffer = 0 ; 

}


GLBuffers::~GLBuffers(){


}

bool GLBuffers::isReady(){
	return vao &&
	        vertex_buffer && 
		normal_buffer && 
		index_buffer && 
		texture_buffer && 
		color_buffer && 
		tangent_buffer && 
		geometry ;  

}

void GLBuffers::clean(){
	if (vao != 0) glDeleteBuffers(1 , &vao); 
	if (vertex_buffer != 0) glDeleteBuffers(1 , &vertex_buffer); 
	if (normal_buffer != 0) glDeleteBuffers(1 , &normal_buffer); 
	if (index_buffer != 0) glDeleteBuffers(1 , &index_buffer); 
	if (texture_buffer != 0) glDeleteBuffers(1 , &texture_buffer); 
	if (color_buffer != 0) glDeleteBuffers(1 , &color_buffer); 
	if (tangent_buffer != 0) glDeleteBuffers(1 , &tangent_buffer); 
}

void GLBuffers::initializeBuffers(){
	glGenVertexArrays(1 , &vao);
	bindVao();
	glGenBuffers(1 , &vertex_buffer); 
	glGenBuffers(1 , &normal_buffer); 
	glGenBuffers(1 , &color_buffer); 
	glGenBuffers(1 , &texture_buffer); 
	glGenBuffers(1 , &tangent_buffer); 
	glGenBuffers(1 , &index_buffer); 
}



void GLBuffers::bindVao(){
	glBindVertexArray(vao) ; 
}

void GLBuffers::bindVertexBuffer(){ 
	glBindBuffer(GL_ARRAY_BUFFER , vertex_buffer); 
} 

void GLBuffers::bindNormalBuffer(){
	glBindBuffer(GL_ARRAY_BUFFER , normal_buffer); 
} 
void GLBuffers::bindTextureBuffer(){ 
	glBindBuffer(GL_ARRAY_BUFFER , texture_buffer); 
} 
void GLBuffers::bindColorBuffer(){ 
	glBindBuffer(GL_ARRAY_BUFFER , color_buffer); 
} 
void GLBuffers::bindIndexBuffer(){ 
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer); 
}

void GLBuffers::bindTangentBuffer(){
	glBindBuffer(GL_ARRAY_BUFFER , tangent_buffer); 
}

void GLBuffers::unbindVao(){
	glBindVertexArray(0) ; 
}

void GLBuffers::fillBuffers(){
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
}


