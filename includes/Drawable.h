#ifndef DRAWABLE_H
#define DRAWABLE_H


#include "Mesh.h"
#include "TextureGroup.h" 
#include "Camera.h"
#include "GLBuffers.h"
#include <QOpenGLWidget>
#include <QDebug>
#include <QString> 




/***
 * \brief OpenGL structures relative to drawing one mesh
 * Manages API calls
 */

class Drawable{
public:
	Drawable();
	Drawable(axomae::Mesh &mesh); 

	virtual ~Drawable(); 
	bool initialize();
	void start_draw(); 
	void end_draw(); 
	void clean();
	void bind();
	void unbind();
	bool ready();
	void setSceneCameraPointer(Camera* camera); 
public:
	axomae::Mesh *mesh_object ;

private:	
	Camera *camera_pointer ; 
	GLBuffers gl_buffers ; 
};











#endif 
