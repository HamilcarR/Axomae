#ifndef SHADER_H
#define SHADER_H

#include "utils_3D.h" 
#include "Material.h" 
#include "Texture.h"
#include "DebugGL.h" 
#include "Camera.h" 
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
	void setMatrixUniform(glm::mat4 matrix_value , const char* uniform_name); 
	void setMatrixUniform(glm::mat3 matrix_value , const char* uniform_name); 
	void setSceneCameraPointer(Camera *camera);  
	virtual void updateCamera(); 
protected:
	virtual void setTextureUniforms();
	template <class T> void setUniformValue(int location , T value); 

private:
	QOpenGLShaderProgram *shader_program; 	
	Camera* camera_pointer; 


};






#endif 
