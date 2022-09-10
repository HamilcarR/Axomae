#ifndef GLVIEWER_H
#define GLVIEWER_H

#include "constants.h"
#include "utils_3D.h" 

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_3_Core> 
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject> 



class QOpenGLShaderProgram ; 

class GLViewer : public QOpenGLWidget , protected QOpenGLFunctions_4_3_Core {

	Q_OBJECT

	public:
		GLViewer(QWidget* parent = nullptr); 
		virtual ~GLViewer();	
		virtual void setNewScene(axomae::Object3D* new_scene);   

	protected:
		void initializeGL() override ; 
		void paintGL() override ; 
		void resizeGL(int width , int height) override ; 
		void printInfo() ; 
	private:
		QOpenGLShaderProgram *shader_program ; 	
		QOpenGLVertexArrayObject vao ; 
		QOpenGLBuffer vertex_buffer ; 
		QOpenGLBuffer normal_buffer ; 
		QOpenGLBuffer index_buffer ; 
		QOpenGLBuffer texture_buffer ;
		QOpenGLBuffer color_buffer ;
		axomae::Object3D *current_scene;
		bool start_draw ; 
		
};




#endif
