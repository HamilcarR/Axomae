#ifndef SHADER_H
#define SHADER_H

#include "utils_3D.h" 
#include "Material.h" 
#include "Texture.h"
#include "DebugGL.h" 
#include "Camera.h" 

class Shader{
public:
	enum TYPE : signed {GENERIC = 0 , BLINN = 1 , CUBEMAP = 2 , PBR = 3};  
	Shader();
	Shader(const std::string vertex_code , const std::string fragment_code); 
	virtual ~Shader();
	virtual void initializeShader();
	virtual void bind(); 
	virtual void release();
	virtual void clean();
	virtual void setShadersRawText(std::string vs , std::string fs) { fragment_shader_txt = fs ; vertex_shader_txt = vs ; } 	
	virtual void updateCamera(); 
	void setModelMatrixUniform(const glm::mat4& matrix);
	void setModelViewProjection(const glm::mat4& model) ; 
	void setModelViewProjection(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model); 
	void enableAttributeArray(GLuint att);
	void setAttributeBuffer(GLuint location , GLenum type , int offset , int tuplesize , int stride = 0 ); 
	void setMatrixUniform(const char* uniform_name , const glm::mat4 &value); 
	void setMatrixUniform(const char* uniform_name , const glm::mat3 &value); 
	void setSceneCameraPointer(Camera *camera);  
	template<typename T> 
	void setUniform(const char* name , const T value) ; 
	
protected:
	virtual void setTextureUniforms();
	void setUniformValue(int location , const int value);
	TYPE type ; 
	unsigned int shader_program; 	
	unsigned int fragment_shader ; 
	unsigned int vertex_shader ;
	std::string fragment_shader_txt ; 
	std::string vertex_shader_txt ; 
	Camera* camera_pointer; 


};

/***********************************************************************************************************************************************************/
class BlinnPhongShader : public Shader{ 
public:
	BlinnPhongShader(); 
	BlinnPhongShader(const std::string vertex_code, const std::string fragment_code) ; 
	virtual ~BlinnPhongShader(); 	

};


/***********************************************************************************************************************************************************/
class CubeMapShader : public Shader {
public:
	CubeMapShader(); 
	CubeMapShader(const std::string vertex_code , const std::string fragment_code); 
	virtual ~CubeMapShader(); 

};

























#endif 
