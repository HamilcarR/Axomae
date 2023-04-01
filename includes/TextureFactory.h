#ifndef TEXTUREFACTORY_H
#define TEXTUREFACTORY_H

#include "Texture.h"

class TextureFactory{
public:
	TextureFactory();
	virtual ~TextureFactory(); 
	static Texture* constructTexture(TextureData* data , Texture::TYPE type ); 	

};



#endif 
