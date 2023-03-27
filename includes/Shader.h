#ifndef SHADER_H
#define SHADER_H

#include "utils_3D.h" 
#include "Material.h" 
#include "Texture.h"
#include <QOpenGLShaderProgram>

class Shader{
public:
	Shader(); 
	virtual ~Shader();
	virtual void initializeShader();
	virtual void bind(); 
	virtual void release();
	virtual void clean(); 
	void enableAttributeArray(GLuint att);
	void setAttributeBuffer(GLuint location , GLenum type , int offset , int tuplesize , int stride = 0 ); 
protected:
	virtual void setTextureUniforms(); 

private:
	QOpenGLShaderProgram *shader_program; 	



};






#endif 
