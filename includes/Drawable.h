#ifndef DRAWABLE_H
#define DRAWABLE_H

#include "Mesh.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_3_Core> 
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject> 
#include <QDebug>
#include <QString> 
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture> 



/***
 * \brief OpenGL structures relative to drawing one mesh
 * Manages API calls
 */

class Drawable{
public:
	Drawable();
	Drawable(axomae::Mesh* mesh); 

	virtual ~Drawable(); 
	bool initialize();
	void start_draw(); 
	void end_draw(); 
	void clean();
	void bind();
	void unbind();
	bool ready();
	
	axomae::Mesh *mesh_object ; 
	
	QOpenGLShaderProgram *shader_program ; 	
	QOpenGLVertexArrayObject vao ; 
	QOpenGLBuffer vertex_buffer ; 
	QOpenGLBuffer normal_buffer ; 
	QOpenGLBuffer index_buffer ; 
	QOpenGLBuffer texture_buffer ;
	QOpenGLBuffer color_buffer ;
	unsigned int sampler2D ; 





};



inline void errorCheck(){
	GLenum error = GL_NO_ERROR;
	do {
    		error = glGetError();
    		if (error != GL_NO_ERROR) 
        		std::cout << "Error:" << error << std::endl ; 
	} while (error != GL_NO_ERROR);

}

inline void shaderErrorCheck(QOpenGLShaderProgram* shader_program){
	if(shader_program){
		QString log = shader_program->log(); 
		std::cout << log.constData() << std::endl; 
	}
}








#endif 
