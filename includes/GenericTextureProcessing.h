#ifndef GENERICTEXTUREPROCESSING_H
#define GENERICTEXTUREPROCESSING_H

#include "utils_3D.h"
#include "constants.h"



/**
 * @brief Class for raw binary data of textures
 * !Note : While using HDR envmap , the data format is still uint32_t , as we wont need to use any other texture format than .hdr files
 * 
 */
class TextureData{
public:
	/**
	 * @brief Rgb channels types 
	 * 
	 */
	enum CHANNELS : unsigned 
	{ 
		RGB = 0 , 
		RGBA = 1
	} ; 	 

	/**
	 * @brief Construct a new Texture Data object
	 * 
	 */
	TextureData(){
		width = 0 ; 
		height = 0 ; 
		data = nullptr ;
		f_data = nullptr ;
		nb_components = 1 ;
		mipmaps = 5 ; 
		internal_format = GL_RGBA ; 
		data_format = GL_BGRA ; 
		data_type = GL_UNSIGNED_BYTE ; 
	}

	/**
	 * @brief Destroy the Texture Data object
	 * 
	 */
	~TextureData(){}

	/**
	 * @brief Copy a texture
	 * 
	 * Provides deep copy of the object , but doesn't do the cleanup for the copied object
	 * 
	 * @param from The texture to be copied 
	 * @return * TextureData& Deep copy of the original TextureData object
	 */
	TextureData& operator=(const TextureData& from){
		if(this != &from){ 
			width = from.width ;
			height = from.height ;
			data_format = from.data_format ; 
			data_type = from.data_type ; 
			internal_format = from.internal_format;
			mipmaps = from.mipmaps ; 
			nb_components = from.nb_components; 
			if(from.data){ 
	    		data = new uint32_t [from.width * from.height] ; 
	    		std::memcpy((void*) data , (void*) from.data , from.width * from.height * sizeof(uint32_t));		
			}
			if(from.f_data){
	    		f_data = new float [from.width * from.height * nb_components] ; 
	    		std::memcpy((void*) data , (void*) from.data , from.width * from.height * nb_components * sizeof(float));		
			}
			name = from.name ; 
		}
		return *this ; 
	}

	/**
	 * @brief Free the object
	 *  
	 */
	void clean(){
		if(data != nullptr)
			delete data ;
		if(f_data)
			delete f_data;  
		data = nullptr ;
		width = 0 ; 
		height = 0 ;
		name = "" ; 
	}

public:
	unsigned int width ;	/**<Width of the texture*/ 
	unsigned int height ; 	/**<Height of the texture*/
	std::string name ; 		/**<Name of the texture*/
	uint32_t *data ; 		/*<1D array raw data of the texture*/
	float *f_data ;         /*<1D float array raw data of the texture (HDR)*/
	unsigned nb_components ; /*<Number of channels*/
	unsigned mipmaps ;       /*<Number of mipmaps*/
	GLenum internal_format ;  
	GLenum data_format ;
	GLenum data_type ;
	
};

/******************************************************************************************************************************************************************************************************************/
class GenericTextureProcessing{
public:
    virtual bool isDimPowerOfTwo(int dim) const = 0 ;
    virtual bool isValidDim(int dim) const = 0 ;
    

};








#endif 