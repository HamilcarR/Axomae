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
class TextureDatabase : public RenderingDatabaseInterface<int , Texture>{
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
	 * @brief Construct a texture of type "type" , and adds it to the database , as a reserved resource or a regular resource.  
	 * @param texture  Raw pixel data
	 * @param type Type of the texture
	 * @param keep_texture_after_clean In case a texture needs to stay between scene changes , this  value needs to be true. 
	 * @param is_dummy Is the texture a dummy texture ?  
	 * @return int Index of the texture inside the database
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
	 * @brief Removes a texture from the database using it's ID
	 * 
	 * @param index 
	 * @return true 
	 * @return false 
	 */
	virtual bool remove(const int index);

	/**
	 * @brief Removes a texture from the database using it's address
	 * 
	 * @param texture Texture to remove 
	 * @return true If the texture has been found
	 * @return false If the address is not in the database
	 */
	virtual bool remove(const Texture* texture);

	/**
	 * @brief Add a texture object to the database . In case the object is already present , this method will return the already present texture's id 
	 * 
	 * @param texture Texture object to add 
	 * @param keep True if texture is to be kept  
	 * @return int Database ID of the texture
	 */
	virtual int add(Texture* texture , bool keep) override; 

	/**
	 * @brief Checks if database contains this index
	 * 
	 * @param index Index to check
	 * @return true If database contains "index"
	 */
	bool contains(const int index) override;

	/**
	 * @brief Checks if a texture is present in the database
	 * 
	 * @param address The texture this methods searches for. 
	 * @return std::pair<int , Texture*> Pair of <ID  , Texture*>. If address is not present in the database , returns < 0 , nullptr> .
	 */
	std::pair<int , Texture*> contains(const Texture* address) override; 

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
