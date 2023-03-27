#ifndef DRAWABLE_H
#define DRAWABLE_H

#include "Mesh.h"
#include "TextureGroup.h" 

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_3_Core> 
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject> 
#include <QDebug>
#include <QString> 




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
};



inline void errorCheck(){
	GLenum error = GL_NO_ERROR;
	do {
    		error = glGetError();
    		if (error != GL_NO_ERROR) 
        		std::cout << "Error:" << error << std::endl ; 
	} while (error != GL_NO_ERROR);

}










#endif 
