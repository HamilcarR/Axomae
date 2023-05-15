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
	 * @brief Removes all elements in the database , destroy them , and free their GL ID 
	 * 
	 */
	void clean();
	
	/**
	 * @brief Construct a new texture if index doesn't exist in the map
	 *  
	 * @param index Unique index of the texture
	 * @param texture Raw texture data 
	 * @param type Type of the texture
	 * @see TextureData
	 * @see Texture::TYPE
	 */
	void addTexture(unsigned int index , TextureData* texture , Texture::TYPE type); 
	
	/**
	 * @brief Get the texture at index
	 * 
	 * @param index Index we want to retrieve
	 * @return Texture* nullptr if nothing found , Texture at "index" else 
	 */
	Texture* get(unsigned int index);
	
	/**
	 * @brief Checks if database contains this index
	 * 
	 * @param index Index to check
	 * @return true If database contains "index"
	 */
	bool contains(unsigned int index);
	
	/**
	 * @brief Retrieve all textures of type "texture_type"
	 * 
	 * @param texture_type Type of the texture 
	 * @return std::vector<std::pair<unsigned int , Texture*>> List of all textures matching "texture_type"
	 * @see Texture
	 * @see Texture::TYPE
	 */
	std::vector<std::pair<unsigned int , Texture*>> getTexturesByType(Texture::TYPE texture_type) const ; 
	
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
	std::map<unsigned int , Texture*> texture_database;		/**<Database of textures*/ 

}; 








#endif 
