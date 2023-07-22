#include "../includes/TextureFactory.h"


TextureFactory::TextureFactory(){

}

TextureFactory::~TextureFactory(){

}


Texture* TextureFactory::constructTexture(TextureData *data , Texture::TYPE type){
	Texture* constructed_texture = nullptr ; 
	switch(type){
		case Texture::DIFFUSE:
			constructed_texture =  data == nullptr ? new DummyDiffuseTexture() : new DiffuseTexture(data); 
		break; 
		case Texture::NORMAL:
			constructed_texture =  data == nullptr ? new DummyNormalTexture() : new NormalTexture(data);
		break ; 
		case Texture::METALLIC:
			constructed_texture = data == nullptr ? new DummyMetallicTexture() : new MetallicTexture(data); 
		break ; 
		case Texture::ROUGHNESS:
			constructed_texture =  data == nullptr ? new DummyRoughnessTexture() : new RoughnessTexture(data); 
		break ; 
		case Texture::AMBIANTOCCLUSION:
			constructed_texture =  data == nullptr ? new DummyAmbiantOcclusionTexture()  : new AmbiantOcclusionTexture(data); 
		break ;
		case Texture::SPECULAR:
			constructed_texture = data == nullptr ? new DummySpecularTexture() : new SpecularTexture(data); 
		break ;
		case Texture::EMISSIVE: 
			constructed_texture = data == nullptr ? new DummyEmissiveTexture() : new EmissiveTexture(data);
		break ; 
		case Texture::OPACITY:
			constructed_texture = data == nullptr ? new DummyOpacityTexture() : new OpacityTexture(data);
		break; 
		case Texture::CUBEMAP:
			constructed_texture =  new CubeMapTexture(data); 
		break ;
		case Texture::ENVMAP:
			constructed_texture =  new EnvironmentMapTexture(data); 
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
