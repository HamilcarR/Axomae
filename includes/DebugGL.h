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

inline void glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,GLsizei length, const GLchar* message, const void* userParam){ 
    std::cout << "OpenGL Debug Message:"
    << "Source:" << source << "\n"
    << "Type:" << type << "\n" 
    << "ID:" << id << "\n" 
    << "Severity:" << severity << "\n" 
    << "Message:" << message << "\n";
}

class GLDebugLogs{

};


#endif 
