#ifndef GLVIEWER_H
#define GLVIEWER_H

#include "constants.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions> 

class GLViewer : public QOpenGLWidget , protected QOpenGLFunctions {

	Q_OBJECT

	public:
		GLViewer(QWidget* parent = nullptr) : QOpenGLWidget{parent}{}
		~GLViewer(){}	
	protected:
		void initializeGL() override ; 
		void paintGL() override ; 
		void resizeGL(int width , int height) override ; 

	private:
		
};




#endif
