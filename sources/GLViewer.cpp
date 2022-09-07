#include "../includes/GLViewer.h" 


void GLViewer::initializeGL() {
	initializeOpenGLFunctions(); 
	glEnable(GL_DEPTH_TEST); 
	glClearColor(1.f , 1.f , 1.f , 1.f); 
}



void GLViewer::paintGL(){


}

void GLViewer::resizeGL(int width , int height){



}
