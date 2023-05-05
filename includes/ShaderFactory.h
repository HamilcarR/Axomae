#ifndef SHADERFACTORY_H
#define SHADERFACTORY_H
#include "Shader.h"

class ShaderFactory{
public:
	ShaderFactory(); 
	virtual ~ShaderFactory(); 
	static Shader* constructShader(std::string vertex_code , std::string fragment_code , Shader::TYPE type); 



}; 






















#endif 
