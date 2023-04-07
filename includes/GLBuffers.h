#ifndef GLBUFFERS_H
#define GLBUFFERS_H

#include "utils_3D.h" 

class GLBuffers{
public:
	GLBuffers();
	GLBuffers(const axomae::Object3D *geometry) ; 
	virtual ~GLBuffers();
	void setGeometryPointer(const axomae::Object3D *geo){geometry = geo;} ; 
	void initializeBuffers();
	bool isReady(); 
	void clean(); 
	void bindVao();
	void unbindVao(); 
	void bindVertexBuffer(); 
	void bindNormalBuffer(); 
	void bindTextureBuffer(); 
	void bindColorBuffer(); 
	void bindIndexBuffer();
	void bindTangentBuffer();
	void fillBuffers(); 
private:
	GLuint vao ; 
	GLuint vertex_buffer ; 
	GLuint normal_buffer ; 
	GLuint index_buffer ; 
	GLuint texture_buffer ;
	GLuint color_buffer ;
	GLuint tangent_buffer ; 
	const axomae::Object3D *geometry; 


}; 







#endif
