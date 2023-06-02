#ifndef TEXTUREGROUP_H
#define TEXTUREGROUP_H

#include "Texture.h" 
#include "TextureDatabase.h"


/**
 * @file TextureGroup.h
 * File defining the TextureGroup structure
 */


/**
 * @class TextureGroup 
 * @brief A TextureGroup object packs a group of different textures to be bound by the shader
 */
class TextureGroup {
public:
	
	/**
	 * @brief Construct a new Texture Group object
	 * 
	 */
	TextureGroup(); 
	
	/**
	 * @brief Destroy the Texture Group object
	 * 
	 */
	virtual ~TextureGroup(); 
	
	/**
	 * @brief Adds a texture to the collection 
	 * 
	 * @param texture_database_index Database index of the texture
	 * @param type Type of the texture
	 */
	virtual void addTexture(int texture_database_index , Texture::TYPE type);
	
	/**
	 * @brief Initialize texture related GL functions and sets up corresponding uniforms
	 * 
	 */
	virtual void initializeGlTextureData(Shader* shader);

	/**
	 * @brief Sets the "initialized" variable to false . 
	 * The cleaning of the textures is done by the TextureDatabase class
	 * @see TextureDatabase
	 * 
	 */
	virtual void clean(); 
	
	/**
	 * @brief Binds every texture in the collection
	 * 
	 */
	virtual void bind(); 
	
	/**
	 * @brief Unbinds every texture in the collection 
	 * 
	 */
	virtual void unbind();
	
	/**
	 * @brief Returns the current state of the texture group
	 * 
	 * @return true if the textures are ready to use
	 * @return false if the textures are not ready to use
	 */
	bool isInitialized(){return initialized;}; 
private:
	std::vector<Texture*> texture_collection; /**<Array of Pointers to textures in the texture database*/
	bool initialized; 						  /**<State of the textures*/ 
	TextureDatabase* texture_database ; 	  /**<Pointer to the database texture*/
};


#endif
