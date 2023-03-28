#include "../includes/Drawable.h"


using namespace axomae ;


Drawable::Drawable(){
	mesh_object = nullptr; 
}

Drawable::Drawable(Mesh &mesh){
	mesh_object = new Mesh(mesh); 
	initialize() ; 
}

Drawable::~Drawable(){

}

void Drawable::clean(){
	mesh_object->clean(); 	
	if(vao.isCreated()){
		vao.release(); 
		vao.destroy(); 
	}
	vertex_buffer.destroy() ; 
	normal_buffer.destroy() ; 
	index_buffer.destroy() ; 
	texture_buffer.destroy() ;
}


bool Drawable::ready(){
	bool vao_created = vao.isCreated() ; 
	bool buffers_created = vertex_buffer.isCreated() && 
			       normal_buffer.isCreated() && 
			       index_buffer.isCreated() && 
			       texture_buffer.isCreated() ; //TODO : add color buffer if useful ? 
	return vao_created && buffers_created  ; 
}

bool Drawable::initialize(){
	if(mesh_object == nullptr)
		return false ; 
	mesh_object->initializeGlData(); 
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
	mesh_object->bindShaders(); 
	mesh_object->shader_program.enableAttributeArray(0);
	mesh_object->shader_program.setAttributeBuffer(0 , GL_FLOAT , 0 , 3 , 0 ) ;
	texture_buffer.bind(); 
	mesh_object->shader_program.enableAttributeArray(1) ; 
	mesh_object->shader_program.setAttributeBuffer(1 , GL_FLOAT , 0 , 2 , 0 ) ; 
	vao.release(); 
	mesh_object->releaseShaders();
	vertex_buffer.release();
	index_buffer.release();
	
	/*manage in texture class*/
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
		mesh_object->bindShaders();
		vao.bind();
	}

}

void Drawable::end_draw(){
	vao.release();
	mesh_object->releaseShaders();


}

void Drawable::bind(){
	mesh_object->bindShaders(); 
	vao.bind();
	mesh_object->bindMaterials(); 
}

void Drawable::unbind(){
	vao.release();
	mesh_object->releaseShaders();
}


