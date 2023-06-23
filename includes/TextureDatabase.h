#ifndef TEXTUREDATABASE_H
#define TEXTUREDATABASE_H

#include <map>
#include "Texture.h" 
#include "TextureFactory.h"
#include "RenderingDatabaseInterface.h"
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
class TextureDatabase : public RenderingDatabaseInterface<Texture>{
public:
	
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
	
	/**
	 * @brief Removes all elements from the database , except those marked "keep"
	 *  
	 */
	void clean() override;

	/**
	 * @brief Deletes everything in the database 
	 * 
	 */
	void purge() override; 

	/**
	 * @brief 
	 * 
	 * @param texture 
	 * @param type 
	 * @param keep_texture_after_clean 
	 * @param is_dummy 
	 * @return int 
	 */
	int addTexture(TextureData* texture , Texture::TYPE type , bool keep_texture_after_clean = false , bool is_dummy = false); 

	/**
	 * @brief Get the texture at index
	 * 
	 * @param index Index we want to retrieve
	 * @return Texture* nullptr if nothing found , Texture at "index" else 
	 */
	Texture* get(const int index) override;
	
	/**
	 * @brief Checks if database contains this index
	 * 
	 * @param index Index to check
	 * @return true If database contains "index"
	 */
	bool contains(const int index) override;
	
	/**
	 * @brief Retrieve all textures of type "texture_type"
	 * 
	 * @param texture_type Type of the texture 
	 * @return std::vector<std::pair<unsigned int , Texture*>> List of all textures matching "texture_type"
	 * @see Texture
	 * @see Texture::TYPE
	 */
	std::vector<std::pair<int , Texture*>> getTexturesByType(Texture::TYPE texture_type) ; 
	
private:
	std::map<int , Texture*> texture_database;				/**<Database of textures*/ 
}; 








#endif 
