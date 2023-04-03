#ifndef DEBUGGL_H
#define DEBUGGL_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_3_Core> 
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject> 
#include <QDebug>

inline void errorCheck(){
	GLenum error = GL_NO_ERROR;
    	error = glGetError();
    	if (error != GL_NO_ERROR) 
        	std::cout << "Error:" << error << std::endl ; 

}




#endif 