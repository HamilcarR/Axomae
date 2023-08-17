#ifndef TEXTURE_H
#define TEXTURE_H

#include "constants.h"
#include "utils_3D.h" 

/**
 * @file Texture.h
 * Implementation of the texture classes 
 * 
 */


/******************************************************************************************************************************************************************************************************************/
class Shader; 
class Texture; 
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
	float *f_data ;
	unsigned nb_components ; 
	unsigned mipmaps ; 
	GLenum internal_format ; 
	GLenum data_format ;
	GLenum data_type ;
	
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Texture class
 * 
 */
class Texture{
public:
	
	/**
	 * @brief Internal format of textures
	 * 
	 */
	enum FORMAT : unsigned {
		/*Internal and data formats*/
		RG = GL_RG ,					
		RGBA = GL_RGBA , 				/**<RGBA with 8 bits per channel*/
		BGRA = GL_BGRA , 				/**<BGRA with 8 bits per channel*/
		RGB = GL_RGB , 					/**<RGB with 8 bits per channel*/
		BGR = GL_BGR , 					/**<BGR with 8 bits per channel*/
		RGBA16F = GL_RGBA16F , 			/**<RGBA with 16 bits floating point per channel*/
		RGBA32F = GL_RGBA32F , 			/**<RGBA with 32 bits floating point per channel*/
		RGB16F = GL_RGB16F , 			/**<RGB with 16 bits floating point per channel*/
		RGB32F = GL_RGB32F  ,			/**<RGB with 32 bits floating point per channel*/	
		/*Data type*/
		UBYTE = GL_UNSIGNED_BYTE ,		 /**<Unsigned byte*/
		FLOAT = GL_FLOAT 				 /**<4 byte float*/
	};

	/**
	 * @brief 
	 * 
	 */
	enum TYPE : signed 
	{
		GENERIC_CUBE = -3 ,					/**<Generic cubemap for general purpose */
		GENERIC = -2 , 						/**<Generic texture used for general purpose */	
		EMPTY = -1 ,						/**<Designate an empty , non generated texture*/ 	
		FRAMEBUFFER = 1,					/**<A texture to be rendered and displayed as a custom framebuffer , by the screen*/
		DIFFUSE = 2 ,						/**<Diffuse texture. In case the shader used is PBR , this is the albedo*/ 
		NORMAL = 3 , 						/**<A normal map texture. Stores normal data*/
		METALLIC = 4 ,						/**<Metallic texture. Stores the amount of metallic property at a given texel.*/ 
		ROUGHNESS = 5 , 					/**<A roughness texture.*/
		AMBIANTOCCLUSION = 6 , 				/**<Ambiant occlusion texture. Occludes light contributions in some areas of the mesh */
		SPECULAR =  7,						/**<Specular texture. In case of PBR , this texture may not be used*/ 
		EMISSIVE = 8 , 						/**<Emissive texture. This texture emits light */
		OPACITY = 9 ,  						/**<Alpha blending map . Provides transparency data*/		
		CUBEMAP = 10 ,						/**<Environment map , in the form of a cubemap. Possesses mip maps for use in specular BRDF*/
		ENVMAP2D = 11 ,						/**<Raw 2D environment map , in equirectangular form. This texture is not used in the final draw loop , until it has been baked into a regular cubemap. */
		IRRADIANCE = 12 ,					/**<Irradiance map . Provides ambient lighting data to the PBR shaders. */
		BRDFLUT = 13 						/**<BRDF lookup texture. Stores reflection factor according to it's texture coordinates*/
		};

	/**
	 * @brief Construct a new empty Texture object
	 * 
	 */
	Texture(); 
	
	/**
	 * @brief Construct a new Texture object from texture data
	 * 
	 * @param tex 
	 */
	Texture(TextureData *tex); 
	
	/**
	 * @brief Destroy the Texture object
	 * 
	 */
	virtual ~Texture();
	
	/**
	 * @brief Sets the raw data 
	 * @param texture A pointer to a TextureData object that contains information about the texture,
	 * including its width, height, and pixel data.
	 */	
	virtual void set(TextureData *texture); 
	
	/**
	 * @brief Cleans the texture
	 * 
	 */
	void clean();

	/**
	 * @brief Get the Texture ID
	 * 
	 * @return unsigned int 
	 */
	unsigned int getSamplerID(){return sampler2D;}

	/**
	 * @brief Set the texture's sampler ID . 
	 * This method will not check if sampler2D has already a valid value.
	 * In this case , the caller needs to free the sampler2D ID first. 
	 * @param id New Sampler2D id. 
	 */
	void setSamplerID(unsigned int id){sampler2D = id;}
	
	/**
	 * @brief Set the Texture Type object
	 * 
	 * @param type Type of the texture
	 * @see Texture::TYPE
	 */
	void setTextureType(TYPE type){name = type;} 	
	
	/**
	 * @brief Get the Texture Type 
	 * 
	 * @return TYPE Type of the texture
	 * @see Texture::TYPE
	 */
	TYPE getTextureType(){return name;} ;  

	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture() = 0 ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture() = 0;
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader) = 0 ;

	/**
	 * @brief Release Opengl data 
	 * 
	 */
	void cleanGlData(); 

	/**
	 * @brief Set the New Size of the texture
	 * 
	 * @param width 
	 * @param height 
	 */
	virtual void setNewSize(unsigned width , unsigned height) ;

	/**
	 * @brief Checks if the texture is a dummy 
	 * @return True if texture is dummy 
	 */
	virtual bool isDummyTexture(){return is_dummy ; }

	/**
	 * @brief Checks if the texture has raw pixel data stored  
	 * 
	 */
	virtual bool hasRawData(){return data != nullptr ;}

	/**
	 * @brief Checks if the texture has been initialized  
	 * 
	 */
	virtual bool isInitialized(){return sampler2D != 0 ; }

	/**
	 * @brief Set the number of mipmaps
	 * 
	 * @param level 
	 */
	virtual void setMipmapsLevel(unsigned level){mipmaps = level;}

	/**
	 * @brief Get the mip maps level number
	 * 
	 * @return unsigned int 
	 */
	virtual unsigned int getMipmapsLevel(){return mipmaps; }

	/**
	 * @brief Generate mip maps , and set texture filters accordingly (LINEAR_MIPMAP_LINEAR)
	 * 
	 */
	virtual void generateMipmap(); 
protected:

	/**
	 * @brief Initialize texture filters , mipmaps and glTexImage2D
	 * 
	 */
	virtual void initializeTexture2D();

	/**
	 * @brief Set the Texture Parameters Options object
	 * 
	 */
	virtual void setTextureParametersOptions();


protected:
	TYPE name ;					/**<Type of the texture*/
	FORMAT internal_format ;	/**<Data layout format on the GPU*/ 
	FORMAT data_format ; 		/**<Raw texture data format*/
	FORMAT data_type ; 
	unsigned int width ; 		/**<Width of the texture*/
	unsigned int height ; 		/**<Height of the texture*/
	uint32_t *data ; 			/**<Raw data of the texture*/
	float *f_data ; 
	unsigned int sampler2D ; 	/**<ID of the texture*/
	unsigned int mipmaps ;		/**<Texture mipmaps level*/ 
	bool is_dummy; 				/**<Check if the current texture is a dummy texture*/
	bool is_initialized ; 
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Diffuse Texture implementation
 * 
 */
class DiffuseTexture : public Texture{
public:

	/**
	 * @brief Construct a new Diffuse Texture object
	 * 
	 */
	DiffuseTexture();
	
	/**
	 * @brief Construct a new Diffuse Texture object from a TextureData object
	 * 
	 * @param data Pointer on a TextureData object
	 */
	DiffuseTexture(TextureData *data); 

	/**
	 * @brief Destroy the Diffuse Texture object
	 * 
	 */
	virtual ~DiffuseTexture();
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture() ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture() ;
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader)  ;

	/**
	 * @brief This overriden method will additionally check for the presence of transparency in the map. 
	 * If alpha < 1.f , the texture is considered as having transparency values. 
	 * 
	 * @param texture Texture data to copy.
	 */
	virtual void set(TextureData *texture) override;

	/**
	 * @brief Returns true if the texture contains alpha value < 1 
	 * 
	 */
	virtual bool hasTransparency(){return has_transparency;}

	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr()   ; 

protected:
	bool has_transparency; 		
}; 

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Normal texture class definition
 * 
 */
class NormalTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Normal Texture object
	 * 
	 */
	NormalTexture();
	
	/**
	 * @brief Construct a new Normal Texture object
	 * 
	 * @param data Raw texture data
	 * @see TextureData
	 */
	NormalTexture(TextureData* data) ; 
	
	/**
	 * @brief Destroy the Normal Texture object
	 * 
	 */
	virtual ~NormalTexture(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture(); 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture() ;
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader)  ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr()   ; 
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Metallic texture class definition 
 * 
 */
class MetallicTexture : public Texture{
public:

	/**
	 * @brief Construct a new Metallic Texture object
	 * 
	 */
	MetallicTexture();
	
	/**
	 * @brief Construct a new Metallic Texture object
	 * 
	 * @param data Raw texture data
	 * @see TextureData 
	 */
	MetallicTexture(TextureData* data); 
	
	/**
	 * @brief Destroy the Metallic Texture object
	 * 
	 */
	virtual ~MetallicTexture(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture() ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture();
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader)  ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr()   ; 
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Roughness texture class definition
 * 
 */
class RoughnessTexture : public Texture{
public:

	/**
	 * @brief Construct a new Roughness Texture object
	 * 
	 */
	RoughnessTexture();
	
	/**
	 * @brief Construct a new Roughness Texture object
	 * 
	 * @param data Raw texture data
	 * @see TextureData
	 */
	RoughnessTexture(TextureData *data); 
	
	/**
	 * @brief Destroy the Roughness Texture object
	 * 
	 */
	virtual ~RoughnessTexture(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture()  ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture();
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader)  ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr() ; 
};


/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Ambiant occlusion texture class definition
 * 
 */
class AmbiantOcclusionTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Ambiant Occlusion Texture object
	 * 
	 */
	AmbiantOcclusionTexture();
	
	/**
	 * @brief Construct a new Ambiant Occlusion Texture object
	 * 
	 * @param data Raw texture data
	 * @see TextureData
	 */
	AmbiantOcclusionTexture(TextureData *data); 
	
	/**
	 * @brief Destroy the Ambiant Occlusion Texture object
	 * 
	 */
	virtual ~AmbiantOcclusionTexture(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture() ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture();
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader)  ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr() ; 	
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Specular texture class definition
 * 
 */
class SpecularTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Specular Texture object
	 * 
	 */
	SpecularTexture(); 
	
	/**
	 * @brief Construct a new Specular Texture object
	 * 
	 * @param data Raw texture data
	 * @see TextureData
	 */
	SpecularTexture(TextureData *data);
	
	/**
	 * @brief Destroy the Specular Texture object
	 * 
	 */
	virtual ~SpecularTexture(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture()  ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture() ;
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader)  ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr() ; 	
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Emissive texture class definition
 * 
 */
class EmissiveTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Emissive Texture object
	 * 
	 */
	EmissiveTexture(); 
	
	/**
	 * @brief Construct a new Emissive Texture object
	 * 
	 * @param data Raw texture data
	 * @see TextureData
	 */
	EmissiveTexture(TextureData *data);

	/**
	 * @brief 
	 * 
	 */
	void initializeTexture2D() override;	
	
	/**
	 * @brief Destroy the Emissive Texture object
	 * 
	 */
	virtual ~EmissiveTexture(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture() ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture();
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader) ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr() ; 	
}; 

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Opacity texture class definition
 * 
 */
class OpacityTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Opacity Texture object
	 * 
	 */
	OpacityTexture(); 
	
	/**
	 * @brief Construct a new Opacity Texture object
	 * 
	 * @param data Raw texture data
	 * @see TextureData
	 */
	OpacityTexture(TextureData *data);
	
	/**
	 * @brief Destroy the Opacity Texture object
	 * 
	 */
	virtual ~OpacityTexture(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture() ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture();
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader) ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr() ; 	
}; 

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Generic texture class definition
 * 
 */
class GenericTexture2D : public Texture{
public:
	
	/**
	 * @brief Construct a new Generic Texture object
	 * 
	 */
	GenericTexture2D();
	
	/**
	 * @brief Construct a new Generic Texture object
	 * 
	 * @param data Raw texture data 
	 * @see TextureData
	 */
	GenericTexture2D(TextureData* data);  
	
	/**
	 * @brief Destroy the Generic Texture object
	 * 
	 */
	virtual ~GenericTexture2D(); 
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture()  ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture();
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader)  ;

	/**
	 * @brief Set the Texture Unit 
	 * 
	 * @param texture_unit 
	 */
	virtual void setTextureUnit(unsigned int texture_unit);

	/**
	 * @brief Set the Location Name object
	 * 
	 * @param name 
	 */
	virtual void setLocationName(std::string name);

	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	const char* getTextureTypeCStr() ; 		

protected:
	unsigned int texture_unit ; 
	std::string location_name ; 

}; 

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Cubemap texture class definition
 * 
 */
class CubemapTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Cube Map Texture object
	 * 
	 */
	CubemapTexture(FORMAT internal_format = RGBA , FORMAT data_format = RGBA , FORMAT data_type = UBYTE , unsigned width = 0 , unsigned height = 0);
	
	/**
	 * @brief Construct a new Cube Map Texture object
	 * 
	 * @param data Texture raw data 
	 * @see TextureData
	 */
	CubemapTexture(TextureData* data);  
	
	/**
	 * @brief Destroy the Cube Map Texture object
	 * 
	 */
	virtual ~CubemapTexture();
	
	/*
	* 
	* width * height is the size of one single face. The total size of the cubemap will be :
	*
	* 	6 x width x height x sizeof(uint32_t) bytes 
	* with height = width .
 	* Here is the layout for mapping the texture : 
	* 
 	*     	  width² = RIGHT => GL_TEXTURE_CUBE_MAP_POSITIVE_X
	* 	  2 x width² = LEFT => GL_TEXTURE_CUBE_MAP_NEGATIVE_X
 	* 	  3 x width² = TOP => GL_TEXTURE_CUBE_MAP_POSITIVE_Y
 	* 	  4 x width² = BOTTOM => GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
 	* 	  5 x width² = BACK => GL_TEXTURE_CUBE_MAP_POSITIVE_Z
 	* 	  6 x width² = FRONT => GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
	*!Note : If TextureData == nullptr , this will instead allocate an empty cubemap . 
	*/
	/**
	 * @brief Initializes the cubemap texture data
	 * 
	 */
	virtual void initializeTexture2D() override; 	
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture()  ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture() ;

	/**
	 * @brief Set the New Size object
	 * 
	 * @param _width 
	 * @param _height 
	 */
	virtual void setNewSize(unsigned _width , unsigned _height) override ; 	
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader) ;

	/**
	 * @brief Generate mipmaps for the cubemap  
	 * 
	 */
	virtual void generateMipmap() override;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr() ;

protected:
	/**
	 * @brief Initialize the cubemap texture data
	 * 
	 * @param texture 
	 */
	virtual void setCubeMapTextureData(TextureData* texture);

}; 


/******************************************************************************************************************************************************************************************************************/
class GenericCubemapTexture : public CubemapTexture{
public:
	/**
	 * @brief Construct a new generic cube Map Texture object
	 * 
	 */
	GenericCubemapTexture(FORMAT internal_format = RGBA , FORMAT data_format = RGBA , FORMAT data_type = UBYTE , unsigned width = 0 , unsigned height = 0);
	
	/**
	 * @brief Construct a new Cube Map Texture object
	 * 
	 * @param data Texture raw data 
	 * @see TextureData
	 */
	GenericCubemapTexture(TextureData* data);  
	
	/**
	 * @brief Destroy the Generic Cubemap Texture object
	 * 
	 */
	virtual ~GenericCubemapTexture();
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture()  ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture() ;


	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader) ;

	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	const char* getTextureTypeCStr() ;

	void setTextureUnit(unsigned int tex_unit){texture_unit = tex_unit;}

	void setLocationName(std::string loc_name){location_name = loc_name;}

	unsigned int getTextureUnit(){return texture_unit;}

	std::string getLocationName(){return location_name;}
	
protected:
	unsigned int texture_unit ; 
	std::string location_name ; 


};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Irradiance texture class definition
 * 
 */
class IrradianceTexture : public CubemapTexture{
public:
	
	/**
	 * @brief Construct an Irradiance cube map 
	 * 
	 */
	IrradianceTexture(FORMAT internal_format = RGB16F , FORMAT data_format = RGB , FORMAT data_type = FLOAT , unsigned width = 0 , unsigned height = 0);
	
	/**
	 * @brief Construct a new Cube Map Texture object
	 * 
	 * @param data Texture raw data 
	 * @see TextureData
	 */
	IrradianceTexture(TextureData* data);  
	
	/**
	 * @brief Destroy the Cube Map Texture object
	 * 
	 */
	virtual ~IrradianceTexture();
	
	/**
	 * @brief Get the texture's alias
	 * 
	 * @return const char* 
	 */
	const char* getTextureTypeCStr() ; 
}; 



/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Environment map texture class definition
 * 
 */
class EnvironmentMap2DTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Cube Map Texture object
	 * 
	 */
	EnvironmentMap2DTexture(FORMAT internal_format = RGB32F , FORMAT data_format = RGB , FORMAT data_type = FLOAT , unsigned width = 0 , unsigned height = 0);
	
	/**
	 * @brief Construct a new environment map Texture object
	 * 
	 * @param data Texture raw data 
	 * @see TextureData
	 */
	EnvironmentMap2DTexture(TextureData* data);  
	
	/**
	 * @brief Destroy the envmap texture
	 * 
	 */
	virtual ~EnvironmentMap2DTexture();
	
	/**
	 * @brief 
	 * 
	 */
	virtual void initializeTexture2D() override; 	
	
	/**
	 * @brief Bind the texture using glBindTexture
	 * 
	 */
	virtual void bindTexture()  ; 

	/**
	 * @brief Unbind texture 
	 * 
	 */
	virtual void unbindTexture() ;
	
	/**
	 * @brief Set the OpenGL texture data infos
	 * 
	 */
	virtual void setGlData(Shader* shader) ;
	
	/**
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr() ;


}; 
/******************************************************************************************************************************************************************************************************************/
/**
 * @class FrameBufferTexture
 * @brief A custom framebuffer's texture for post processing
 * 
 */
class FrameBufferTexture : public Texture {
public:
	
	/**
	 * @brief Construct a new Frame Buffer Texure object
	 * 
	 */
	FrameBufferTexture();

	/**
	 * @brief Construct a new Frame Buffer Texture
	 * Contains only width , and height... The rest of the TextureData parameter is not used,
	 * @param data TextureData parameter 
	 * 
	 */
	FrameBufferTexture(TextureData* data ); 

	/**
	 * @brief Construct a new Frame Buffer Texture 
	 * 
	 * @param width Width of the texture
	 * @param height Height of the texture
	 */
	FrameBufferTexture(unsigned width , unsigned height);  
	
	/**
	 * @brief Destroy the Frame Buffer Texure object
	 * 
	 */
	virtual ~FrameBufferTexture(); 
	
	/**
	 * @brief 
	 * 
	 */
	virtual void bindTexture() ;
	
	/**
	 * @brief 
	 * 
	 */
	virtual void unbindTexture();
	
	/**
	 * @brief Set the Gl Data object
	 * 
	 */
	virtual void setGlData(Shader* shader); 

	/**
	 * @brief 
	 * 
	 * @return * void 
	 */
	virtual void initializeTexture2D() override ;

	/**
	 * @brief Get the Texture string name
	 * 
	 * @return const char* 
	 */
	static const char* getTextureTypeCStr(); 

protected:

}; 

/******************************************************************************************************************************************************************************************************************/

/**
 * @brief Pre baked texture storing the amount of specular reflection for a given triplets of (I , V , R) , where I is the incident light , V is the view direction , and R a roughness value . IE , for (X , Y) being the
 * texture coordinates , Y is a roughness scale , X <==> (I dot V) is the angle betweend the incident light and view direction.
 * @class BRDFLookupTexture
 */
class BRDFLookupTexture : public Texture{
public:
	/**
	 * @brief Construct a new BRDFLookupTexture object
	 * 
	 */
	BRDFLookupTexture();
	
	/**
	 * @brief Construct a new BRDFLookupTexture object
	 * 
	 * @param data 
	 */
	BRDFLookupTexture(TextureData* data);
	
	/**
	 * @brief Destroy the BRDFLookupTexture object
	 * 
	 */
	virtual ~BRDFLookupTexture(); 
	
	/**
	 * @brief Binds the texture
	 * 
	 */
	virtual void bindTexture(); 
	
	/**
	 * @brief Unbinds the texture
	 * 
	 */
	virtual void unbindTexture(); 
	
	/**
	 * @brief Initializes the texture and gives it it's ID 
	 * 
	 * @param shader 
	 */
	virtual void setGlData(Shader* shader); 
	
	/**
	 * @brief Initializes the filters of the texture 
	 * 
	 */
	virtual void initializeTexture2D() override ; 
	
	/**
	 * @brief Get the Texture Type C Str object
	 * 
	 * @return const char* 
	 */
	static const char* getTextureTypeCStr(); 
protected:

};


/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummyDiffuseTexture
 * @brief Class implementing an empty diffuse texture 
 */
class DummyDiffuseTexture:public DiffuseTexture{
public: 
	DummyDiffuseTexture(); 
	virtual ~DummyDiffuseTexture(); 

};

/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummyNormalTexture
 * @brief Class implementing an empty normal texture
 * 
 */
class DummyNormalTexture:public NormalTexture{
public:
	DummyNormalTexture(); 
	virtual ~DummyNormalTexture(); 

}; 

/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummyMetallicTexture
 * @brief Class implementing an empty Metallic texture
 * 
 */
class DummyMetallicTexture:public MetallicTexture{
public:
	DummyMetallicTexture(); 
	virtual ~DummyMetallicTexture(); 

}; 

/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummyRoughnessTexture
 * @brief Class implementing an empty Roughness texture
 * 
 */
class DummyRoughnessTexture:public RoughnessTexture{
public:
	DummyRoughnessTexture(); 
	virtual ~DummyRoughnessTexture(); 

}; 

/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummyAmbiantOcclusionTexture
 * @brief Class implementing an empty Ambiant Occlusion texture
 * 
 */
class DummyAmbiantOcclusionTexture:public AmbiantOcclusionTexture{
public:
	DummyAmbiantOcclusionTexture(); 
	virtual ~DummyAmbiantOcclusionTexture(); 

}; 

/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummySpecularTexture
 * @brief Class implementing an empty Specular texture
 * 
 */
class DummySpecularTexture:public SpecularTexture{
public:
	DummySpecularTexture(); 
	virtual ~DummySpecularTexture(); 

}; 

/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummyEmissiveTexture
 * @brief Class implementing an empty Emissive texture
 * 
 */
class DummyEmissiveTexture:public EmissiveTexture{
public:
	DummyEmissiveTexture(); 
	virtual ~DummyEmissiveTexture(); 

}; 

/******************************************************************************************************************************************************************************************************************/

/**
 * @class DummyOpacityTexture
 * @brief Class implementing an empty Opacity texture
 * 
 */
class DummyOpacityTexture:public OpacityTexture{
public:
	DummyOpacityTexture(); 
	virtual ~DummyOpacityTexture(); 

}; 








#endif 
