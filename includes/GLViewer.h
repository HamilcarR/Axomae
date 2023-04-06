#ifndef GLVIEWER_H
#define GLVIEWER_H

#include <QMouseEvent> 

#include "constants.h"
#include "utils_3D.h" 
#include "Renderer.h" 
#include "Mesh.h"




/**
 * @brief Class responsible for the 3D rendering
 */

class QOpenGLShaderProgram ; 
class QOpenGLTexture ; 
class GLViewer : public QOpenGLWidget , protected QOpenGLFunctions_4_3_Core {

	Q_OBJECT
	
	public:
		GLViewer(QWidget* parent = nullptr); 
		virtual ~GLViewer();	
		virtual void setNewScene(std::vector<axomae::Mesh> &new_scene);   
	protected:
		void initializeGL() override ; 
		void paintGL() override ; 
		void resizeGL(int width , int height) override ; 
		void printInfo() ; 
	private:
		void mouseMoveEvent(QMouseEvent *event) override; 
		void mousePressEvent(QMouseEvent *event) override; 
		void mouseReleaseEvent(QMouseEvent *event) override; 
		void wheelEvent(QWheelEvent *event) override;
	private:
		Renderer* renderer;	
			






};




#endif
