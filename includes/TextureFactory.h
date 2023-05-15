#ifndef TEXTUREFACTORY_H
#define TEXTUREFACTORY_H

#include "Texture.h"

/**
 * @file TextureFactory.h
 * Class definition of the factory system for the textures 
 * 
 */


/**
 * @class TextureFactory
 * @brief Provides a way to easily create textures from raw data 
 *  
 */
class TextureFactory{
public:

	/**
	 * @brief Construct a new Texture Factory object
	 * 
	 */
	TextureFactory();
	
	/**
	 * @brief Destroy the Texture Factory object
	 * 
	 */
	virtual ~TextureFactory(); 
	
	/**
	 * @brief Construct a new Texture* object from raw texture data , and a type
	 * 
	 * @param data Contains raw data about the texture 
	 * @param type Type of the texture we want to create
	 * @return Texture* The created texture object
	 * @see Texture
	 * @see Texture::TYPE
	 */
	static Texture* constructTexture(TextureData* data , Texture::TYPE type ); 	

};



#endif 
