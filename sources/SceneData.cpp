#include "../includes/SceneData.h"


using namespace axomae ; 




SceneData::SceneData(){
	current_scene = nullptr ; 
	start_draw = false ;	
	shader_program = nullptr ;

}

SceneData::~SceneData(){
	if(shader_program != nullptr){
		std::cout << "Destroying shader : "<< shader_program  << std::endl ;  
		delete shader_program ;
		shader_program = nullptr ; 
	}			
	if(current_scene != nullptr){
		delete current_scene ; 
		current_scene = nullptr ; 
	}
	if(vao.isCreated()){
		vao.release(); 
		vao.destroy(); 
	}
}

void SceneData::initialize(){
	glEnable(GL_DEPTH_TEST);
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
	vertex_buffer.create();
	texture_buffer.create() ; 
	index_buffer.create();
	vertex_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw);	
	texture_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw); 
	index_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
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
	glGenTextures(1 , &sampler2D); 
}

bool SceneData::prep_draw(){
	if(start_draw && shader_program && current_scene){
		/*Bind buffers*/
		vertex_buffer.bind();
		vertex_buffer.allocate(current_scene->geometry.vertices.data() , current_scene->geometry.vertices.size() * sizeof(float)); 
		vertex_buffer.release();
		texture_buffer.bind();
		texture_buffer.allocate(current_scene->geometry.uv.data() , current_scene->geometry.uv.size() * sizeof(float)) ; 
		texture_buffer.release(); 
		index_buffer.bind();
		index_buffer.allocate(current_scene->geometry.indices.data() , current_scene->geometry.indices.size() * sizeof(unsigned int)); 
		index_buffer.release();
		/*draw command*/
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
		shader_program->bind();
		vao.bind();		
		return true; 						
	}
	else{
		glClearColor(0 , 0 , 0, 1.f);
		return false ;	
	}
}

void SceneData::end_draw(){
	vao.release();
	shader_program->release();
}

void SceneData::setNewScene(std::vector<Mesh> &new_scene){
	if(current_scene == nullptr)
		current_scene = new Mesh ;
	*current_scene = new_scene[0];
	start_draw = true ;
	glActiveTexture(GL_TEXTURE0); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	glTexImage2D(GL_TEXTURE_2D , 0 , GL_RGBA , current_scene->material.textures.diffuse.width , current_scene->material.textures.diffuse.height , 0 , GL_BGRA , GL_UNSIGNED_BYTE , current_scene->material.textures.diffuse.data); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	shader_program->setUniformValue(shader_program->uniformLocation("diffuse") , 0) ; 
}


