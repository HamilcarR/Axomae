#include "../includes/ShaderFactory.h"

ShaderFactory::ShaderFactory(){}
ShaderFactory::~ShaderFactory(){}


Shader* ShaderFactory::constructShader(std::string v , std::string f , Shader::TYPE type){
	Shader* constructed_shader = nullptr ; 
	switch(type){
		case Shader::GENERIC: 
			constructed_shader = new Shader(v , f) ; 
		break; 
		case Shader::BLINN:
			constructed_shader = new BlinnPhongShader(v , f) ; 
		break; 
		case Shader::CUBEMAP:
			constructed_shader = new CubeMapShader(v , f) ;
		break; 
		default:
			constructed_shader = nullptr ; 
		break; 
	}
	return constructed_shader ; 
}
