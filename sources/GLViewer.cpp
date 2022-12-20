#include "../includes/GLViewer.h" 

using namespace axomae ; 


GLViewer::GLViewer(QWidget* parent) : QOpenGLWidget (parent) {
	scene_data = new SceneData  ; 	
}

GLViewer::~GLViewer(){
	makeCurrent() ; 
	delete scene_data ; 
	doneCurrent(); 
}

void GLViewer::initializeGL() {
	initializeOpenGLFunctions(); 
	scene_data->initialize() ; 
	errorCheck() ;
}

void GLViewer::paintGL(){
	if(scene_data->prep_draw()){
		errorCheck() ;  
		glDrawElements(GL_TRIANGLES , scene_data->current_scene->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
		scene_data->end_draw(); 
	}

}
void GLViewer::resizeGL(int width , int height){

}
void GLViewer::printInfo(){
	std::cout << "Renderer info here : < >" << std::endl; 	
}
void GLViewer::setNewScene(std::vector<Mesh> &new_scene){
	makeCurrent();
	scene_data->setNewScene(new_scene); 
	doneCurrent();

}

