#ifndef TEXTUREGROUP_H
#define TEXTUREGROUP_H

#include "Texture.h" 


class TextureGroup {
public:
	TextureGroup(); 
	virtual ~TextureGroup(); 
	virtual void addTexture(TextureData &tex , Texture::TYPE type);
	virtual void initializeGlTextureData();
	virtual void clean(); 
	virtual void bind(); 
	virtual void unbind();
	bool isInitialized(){return initialized;}; 
private:
	std::vector<Texture*> texture_collection; 
	bool initialized; 

};


#endif
