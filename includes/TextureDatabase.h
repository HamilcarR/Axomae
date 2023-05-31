#ifndef TEXTUREDATABASE_H
#define TEXTUREDATABASE_H

#include <map>
#include "Texture.h" 
#include "TextureFactory.h"

/**
 * @file TextureDatabase.h
 * Class definition for the texture database 
 * 
 */

/**
 * @brief TextureDatabase class definition
 * The TextureDatabase class is a singleton that holds an std::map with unique key associated with a texture type
 * We use it to keep texture objects in one place , meshes will only reference the textures. 
 *
 */
class TextureDatabase{
public:
	
	/**
	 * @brief Get the unique instance of the database
	 * 
	 * @return TextureDatabase* Unique instance of the database
	 */
	static TextureDatabase* getInstance(); 
 
	/**
	 * @brief Destroy the database
	 * 
	 */
 	void destroy(); 
	
	/**
	 * @brief Removes all elements from the database , except those marked "keep"
	 *  
	 */
	void softCleanse();

	/**
	 * @brief Deletes everything in the database 
	 * 
	 */
	void hardCleanse(); 
	
	/**
	 * @brief Construct a new texture and adds it to the database with a unique index
	 *  
	 * @param texture Raw texture data 
	 * @param type Type of the texture
	 * @param keep_texture_after_clean Some textures could benefit from not being deleted after each scene change, like framebuffer textures
	 * @see TextureData
	 * @see Texture::TYPE
	 */
	int addTexture(TextureData* texture , Texture::TYPE type , bool keep_texture_after_clean); 
	
	/**
	 * @brief Get the texture at index
	 * 
	 * @param index Index we want to retrieve
	 * @return Texture* nullptr if nothing found , Texture at "index" else 
	 */
	Texture* get(int index);
	
	/**
	 * @brief Checks if database contains this index
	 * 
	 * @param index Index to check
	 * @return true If database contains "index"
	 */
	bool contains(int index);
	
	/**
	 * @brief Retrieve all textures of type "texture_type"
	 * 
	 * @param texture_type Type of the texture 
	 * @return std::vector<std::pair<unsigned int , Texture*>> List of all textures matching "texture_type"
	 * @see Texture
	 * @see Texture::TYPE
	 */
	std::vector<std::pair<int , Texture*>> getTexturesByType(Texture::TYPE texture_type) const ; 
	
	/**
	 * @brief Check if database is initialized
	 * 
	 * @return true If database is initialized 
	 */
	static bool isInstanced(){ return instance != nullptr ; } ; 
	
private:

	/**
	 * @brief Construct a new Texture Database object
	 * 
	 */
	TextureDatabase(); 	
	
	/**
	 * @brief Destroy the Texture Database object
	 * 
	 */
	virtual ~TextureDatabase(); 	
	TextureDatabase (const TextureDatabase&) = delete;
  	TextureDatabase& operator=(const TextureDatabase&) = delete;

private:
	static TextureDatabase* instance ;						/**<Unique instance pointer of the database*/			
	std::map<int , Texture*> texture_database;				/**<Database of textures*/ 
}; 








#endif 
