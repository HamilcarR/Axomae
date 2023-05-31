#include "../includes/TextureFactory.h"


TextureFactory::TextureFactory(){

}

TextureFactory::~TextureFactory(){

}


Texture* TextureFactory::constructTexture(TextureData *data , Texture::TYPE type){
	Texture* constructed_texture = nullptr ; 
	switch(type){
		case Texture::DIFFUSE:
			constructed_texture =  new DiffuseTexture(data); 
		break; 
		case Texture::NORMAL:
			constructed_texture =  new NormalTexture(data);
		break ; 
		case Texture::METALLIC:
			constructed_texture =  new MetallicTexture(data); 
		break ; 
		case Texture::ROUGHNESS:
			constructed_texture =  new RoughnessTexture(data); 
		break ; 
		case Texture::AMBIANTOCCLUSION:
			constructed_texture =  new AmbiantOcclusionTexture(data); 
		break ;
		case Texture::SPECULAR:
			constructed_texture =  new SpecularTexture(data); 
		break ;
		case Texture::EMISSIVE: 
			constructed_texture =  new EmissiveTexture(data);
		break ; 
		case Texture::CUBEMAP:
			constructed_texture =  new CubeMapTexture(data); 
		break ;
		case Texture::FRAMEBUFFER:
			constructed_texture = new FrameBufferTexture(data);
		break; 
		case Texture::GENERIC: 
			constructed_texture =  new GenericTexture(data); 
		break ; 
		default : 
			constructed_texture =  nullptr ;
		break ;
	}
	if(constructed_texture)
		constructed_texture->setTextureType(type); 
	return constructed_texture ; 

}
