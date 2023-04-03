#include "../includes/GLViewer.h" 
#include <QPoint>
#include <QCursor>

using namespace axomae ; 


GLViewer::GLViewer(QWidget* parent) : QOpenGLWidget (parent) {
	renderer = new Renderer()  ; 	
}

GLViewer::~GLViewer(){
	makeCurrent() ; 
	delete renderer ; 
	doneCurrent(); 
}

void GLViewer::initializeGL() {
	initializeOpenGLFunctions(); 
	renderer->initialize() ; 
	errorCheck() ;
}

void GLViewer::paintGL(){
	if(renderer->prep_draw()){
		std::cout << "renderer prepped" << std::endl ; 
		errorCheck(); 	
		renderer->draw(dynamic_cast<QOpenGLFunctions_4_3_Core*> (this) ) ; 
		renderer->end_draw(); 
	}
}

void GLViewer::resizeGL(int width , int height){	 
	
}

void GLViewer::printInfo(){
	std::cout << "Renderer info here : < >" << std::endl; 	
}

void GLViewer::mouseMoveEvent(QMouseEvent *event){
	QPoint p = this->mapFromGlobal(QCursor::pos()); 	
	update(); 
}


void GLViewer::setNewScene(std::vector<Mesh> &new_scene){
	makeCurrent();
	renderer->set_new_scene(new_scene); 
	doneCurrent();
}

