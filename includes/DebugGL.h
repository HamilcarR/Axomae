#ifndef DEBUGGL_H
#define DEBUGGL_H

#include <iostream>
#include "utils_3D.h" 
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>


inline void errorCheck(const char* filename , unsigned int line ){
	GLenum error = GL_NO_ERROR;
    error = glGetError();
    if (error != GL_NO_ERROR) 
        std::cerr << "GL error in file : " << filename << " at line : " << line << " with error : " << error << std::endl ; 

}




class GLDebugLogs{

};


#endif 
