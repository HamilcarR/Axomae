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
		case Shader::SCREEN_FRAMEBUFFER:
			constructed_shader = new ScreenFrameBufferShader(v , f); 
		break;
		case Shader::BOUNDING_BOX:
			constructed_shader = new BoundingBoxShader(v , f); 
		break;
		case Shader::PBR:
			constructed_shader = new PBRShader(v , f); 
		break;
		case Shader::ENVMAP_CUBEMAP_CONVERTER:
			constructed_shader = new EnvmapCubemapBakerShader(v , f) ; 
		break;
		case Shader::IRRADIANCE_CUBEMAP_COMPUTE:
			constructed_shader = new IrradianceCubemapBakerShader(v , f); 
		break ; 
		default:
			constructed_shader = nullptr ; 
		break; 
	}
	return constructed_shader ; 
}
