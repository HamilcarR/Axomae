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
	renderer->setScreenSize(width() , height()); 
	renderer->initialize() ; 
	errorCheck() ;
}

void GLViewer::paintGL(){
	

	if(renderer->prep_draw()){
	//	errorCheck(); 	
		renderer->draw(dynamic_cast<QOpenGLFunctions_4_3_Core*> (this) ) ; 
		renderer->end_draw(); 
	}
}

void GLViewer::resizeGL(int w , int h){	 
	renderer->setScreenSize(width() , height()); 
}

void GLViewer::printInfo(){
	std::cout << "Renderer info here : < >" << std::endl; 	
}

void GLViewer::mouseMoveEvent(QMouseEvent *event){
	QPoint p = this->mapFromGlobal(QCursor::pos()); 	
	MouseState* mouse = renderer->getMouseStatePointer();
	mouse->previous_pos_x = mouse->pos_x ; 
	mouse->previous_pos_y = mouse->pos_y ; 
	mouse->pos_x = p.x(); 
	mouse->pos_y = p.y();
	update(); 
}

void GLViewer::mousePressEvent(QMouseEvent *event){
	MouseState *mouse = renderer->getMouseStatePointer();
	switch(event->button()){
		case Qt::LeftButton:
			mouse->left_button_clicked = true; 
			mouse->left_button_released = false ; 
			renderer->onLeftClick(); 
		break; 
		case Qt::RightButton:
			mouse->right_button_clicked = true ; 
			mouse->right_button_released = false ; 
			renderer->onRightClick(); 
		break; 
		default:
		break ; 
	}

	update(); 
}

void GLViewer::mouseReleaseEvent(QMouseEvent* event){
	MouseState *mouse = renderer->getMouseStatePointer();
	switch(event->button()){
		case Qt::LeftButton:
			mouse->left_button_clicked = false; 
			mouse->left_button_released = true ; 
			renderer->onLeftClickRelease(); 
		break; 
		case Qt::RightButton:
			mouse->right_button_clicked = false ; 
			mouse->right_button_released = true ;
			renderer->onRightClickRelease(); 
		break;
		default:
		break; 
	}
	update(); 
}

void GLViewer::setNewScene(std::vector<Mesh> &new_scene){
	makeCurrent();
	renderer->set_new_scene(new_scene); 
	doneCurrent();
}

