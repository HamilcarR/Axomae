#include "../includes/GLViewer.h" 
#include "../includes/Loader.h" 


#include <QDebug>
#include <QString> 
#include <QOpenGLShaderProgram>


using namespace axomae ; 


static inline void errorCheck(){
	GLenum error = GL_NO_ERROR;
	do {
    		error = glGetError();
    		if (error != GL_NO_ERROR) 
        		std::cout << "Error:" << error << std::endl ; 
	} while (error != GL_NO_ERROR);

}

static inline void shaderErrorCheck(QOpenGLShaderProgram* shader_program){
	if(shader_program){
		QString log = shader_program->log(); 
		std::cout << log.constData() << std::endl; 
	}
}

GLViewer::GLViewer(QWidget* parent) : QOpenGLWidget (parent) {
	current_scene = nullptr ; 
	start_draw = false ;	
	shader_program = nullptr ;
}

GLViewer::~GLViewer(){
	makeCurrent() ; 
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
	doneCurrent(); 

}



void GLViewer::initializeGL() {
	initializeOpenGLFunctions(); 
	glEnable(GL_DEPTH_TEST);
	shader_program = new QOpenGLShaderProgram(); 
	shader_program->addShaderFromSourceFile(QOpenGLShader::Vertex , "../shaders/simple.vert"); 
	shader_program->addShaderFromSourceFile(QOpenGLShader::Fragment , "../shaders/simple.frag"); 	
	shader_program->link();
	shaderErrorCheck(shader_program) ; 
	bool vao_okay = vao.create();
	assert(vao_okay == true) ; 
	vertex_buffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer); 	
	vertex_buffer.create();
	index_buffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer); 
	index_buffer.create();
	vertex_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	index_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	vao.bind(); 
	vertex_buffer.bind(); 
	index_buffer.bind();
	shader_program->bind(); 
	shader_program->enableAttributeArray(0);
	shader_program->setAttributeBuffer(0 , GL_FLOAT , 0 , 3 , 0 ) ;
	vao.release(); 
	shader_program->release();
	vertex_buffer.release();
	index_buffer.release(); 
	errorCheck() ;

}

void GLViewer::paintGL(){
	if(start_draw && shader_program && current_scene){
		vertex_buffer.bind();
		vertex_buffer.allocate(current_scene->vertices.data() , current_scene->vertices.size() * sizeof(float)); 
		vertex_buffer.release(); 
		index_buffer.bind();
		index_buffer.allocate(current_scene->indices.data() , current_scene->indices.size() * sizeof(unsigned int)); 
		index_buffer.release();
		{ //draw command
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
			shader_program->bind();
			vao.bind();		
			errorCheck() ;  
			glDrawElements(GL_TRIANGLES , current_scene->indices.size() , GL_UNSIGNED_INT , 0 );
			vao.release();
			shader_program->release();
		}
	}
	else
		glClearColor(0 , 0 , 0, 1.f);
}

void GLViewer::resizeGL(int width , int height){

}

void GLViewer::printInfo(){
	std::cout << "Renderer info here : < >" << std::endl; 	
}

void GLViewer::setNewScene(Object3D* new_scene){
	if(current_scene == nullptr)
		current_scene = new Object3D ;
	*current_scene = *new_scene ;
	start_draw = true ;
}

