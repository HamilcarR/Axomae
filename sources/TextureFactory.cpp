#include "../includes/TextureFactory.h"


TextureFactory::TextureFactory(){

}

TextureFactory::~TextureFactory(){

}


Texture* TextureFactory::constructTexture(TextureData *data , Texture::TYPE type){
	switch(type){
		case Texture::DIFFUSE:
			return new DiffuseTexture(data); 
		break; 
		case Texture::NORMAL:
			return new NormalTexture(data);
		break ; 
		case Texture::METALLIC:
			return new MetallicTexture(data); 
		break ; 
		case Texture::ROUGHNESS:
			return new RoughnessTexture(data); 
		break ; 
		case Texture::AMBIANTOCCLUSION:
			return new AmbiantOcclusionTexture(data); 
		break ;
		case Texture::SPECULARTINT:
			return new SpecularTintTexture(data); 
		break ;
		case Texture::GENERIC: 
			return new GenericTexture(data); 
		break ; 
		default : 
			return nullptr ;
		break ;
	}
}
