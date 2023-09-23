#include "../includes/GLViewer.h"
#include <QPoint>
#include <QCursor>
#include <QSurfaceFormat>

using namespace axomae;

GLViewer::GLViewer(QWidget *parent) : QOpenGLWidget(parent){
	QSurfaceFormat format;	
	format.setRenderableType(QSurfaceFormat::OpenGL); 
	format.setVersion(4, 6);	
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setOption(QSurfaceFormat::DebugContext); 
	format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
	format.setAlphaBufferSize(8); 
	format.setSwapInterval(1);
	setFormat(format);
	renderer = std::make_unique<Renderer>(width() , height() , this);
	glew_initialized = false;
}

GLViewer::~GLViewer(){
}

void GLViewer::initializeGL(){
	makeCurrent();
	if (!glew_initialized){
		glewExperimental = GL_TRUE ; 
		GLenum err = glewInit();
		std::cout << "glew initialized!\n";
		if (err != GLEW_OK)
		{
			std::cerr << "failed to initialize glew with error : " << reinterpret_cast<const char *>(glewGetErrorString(err)) << "\n";
			exit(EXIT_FAILURE);
		}
		else{
			glew_initialized = true;
			if(GLEW_ARB_debug_output){			
				glEnable(GL_DEBUG_OUTPUT); 
				glDebugMessageCallback(glDebugCallback , nullptr); 
			}
			else{
				std::cout << "Debug output extension not supported\n" ; 
			}
		}
	}
	renderer->onResize(width(), height());
	renderer->initialize();
	errorCheck(__FILE__ , __LINE__);
}

void GLViewer::paintGL(){
	if (renderer->prep_draw()){
		renderer->setDefaultFrameBufferId(defaultFramebufferObject()); 
		renderer->draw();		
	}	
}

void GLViewer::resizeGL(int w, int h){
	renderer->onResize(width(), height());
}

void GLViewer::printInfo(){
	std::cout << "Renderer info here : < >" << std::endl;
}

void GLViewer::mouseMoveEvent(QMouseEvent *event){
	QPoint p = this->mapFromGlobal(QCursor::pos());
	MouseState *mouse = renderer->getMouseStatePointer();
	QRect bounds = this->rect();
	if(bounds.contains(p)){
		mouse->previous_pos_x = mouse->pos_x;
		mouse->previous_pos_y = mouse->pos_y;
		mouse->pos_x = p.x();
		mouse->pos_y = p.y();
	}
	update();
}

void GLViewer::wheelEvent(QWheelEvent *event){
	if (event->angleDelta().y() > 0) // scroll up
		renderer->onScrollUp();
	else if (event->angleDelta().y() < 0)
		renderer->onScrollDown();
	update();
}

void GLViewer::mousePressEvent(QMouseEvent *event){
	MouseState *mouse = renderer->getMouseStatePointer();
	switch (event->button()){
	case Qt::LeftButton:
		mouse->left_button_clicked = true;
		mouse->left_button_released = false;
		renderer->onLeftClick();
		break;
	case Qt::RightButton:
		mouse->right_button_clicked = true;
		mouse->right_button_released = false;
		renderer->onRightClick();
		break;
	default:
		break;
	}
	update();
}

void GLViewer::mouseReleaseEvent(QMouseEvent *event){
	MouseState *mouse = renderer->getMouseStatePointer();
	switch (event->button()){
	case Qt::LeftButton:
		mouse->left_button_clicked = false;
		mouse->left_button_released = true;
		renderer->onLeftClickRelease();
		break;
	case Qt::RightButton:
		mouse->right_button_clicked = false;
		mouse->right_button_released = true;
		renderer->onRightClickRelease();
		break;
	default:
		break;
	}
	update();
}

void GLViewer::setNewScene(std::pair<std::vector<Mesh *> , SceneTree> &new_scene){
	makeCurrent();
	renderer->set_new_scene(new_scene);
	doneCurrent();
}

