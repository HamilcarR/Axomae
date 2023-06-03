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
/**
 * @brief Class for raw binary data of textures
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
		width = from.width ;
		height = from.height ; 
		data = new uint32_t [from.width * from.height] ; 
		memcpy((void*) data , (void*) from.data , from.width * from.height * sizeof(uint32_t));		
		name = from.name ; 
		return *this ;
	}

	/**
	 * @brief Free the object
	 *  
	 */
	void clean(){
		if(data != nullptr)
			delete data ; 
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
};

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief Texture class
 * 
 */
class Texture{
public:

	enum FORMAT : signed {
		RGBA = GL_RGBA , 
		BGRA = GL_BGRA , 
		RGB = GL_RGB , 
		BGR = GL_BGR 
	};

	/**
	 * @brief Type of the texture
	 * 
	 */
	enum TYPE : signed 
	{
		EMPTY = -1 , 
		DIFFUSE = 0 , 
		NORMAL = 1 , 
		METALLIC = 2 , 
		ROUGHNESS = 3 , 
		AMBIANTOCCLUSION = 4 , 
		SPECULAR = 5, 
		EMISSIVE = 6 , 
		CUBEMAP = 7 , 
		GENERIC = 8 , 
		FRAMEBUFFER = 9
	} ;

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
	void set(TextureData *texture); 
	
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

	virtual void setNewSize(unsigned width , unsigned height) ;

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
	unsigned int width ; 		/**<Width of the texture*/
	unsigned int height ; 		/**<Height of the texture*/
	uint32_t *data ; 			/**<Raw data of the texture*/
	unsigned int sampler2D ; 	/**<ID of the texture*/
	
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
	 * @brief Get the texture string description
	 * 
	 * @return C string 
	 */
	static const char* getTextureTypeCStr()   ; 	
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
 * @brief Generic texture class definition
 * 
 */
class GenericTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Generic Texture object
	 * 
	 */
	GenericTexture();
	
	/**
	 * @brief Construct a new Generic Texture object
	 * 
	 * @param data Raw texture data 
	 * @see TextureData
	 */
	GenericTexture(TextureData* data);  
	
	/**
	 * @brief Destroy the Generic Texture object
	 * 
	 */
	virtual ~GenericTexture(); 
	
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
 * @brief Cubemap texture class definition
 * 
 */
class CubeMapTexture : public Texture{
public:
	
	/**
	 * @brief Construct a new Cube Map Texture object
	 * 
	 */
	CubeMapTexture();
	
	/**
	 * @brief Construct a new Cube Map Texture object
	 * 
	 * @param data Texture raw data 
	 * @see TextureData
	 */
	CubeMapTexture(TextureData* data);  
	
	/**
	 * @brief Destroy the Cube Map Texture object
	 * 
	 */
	virtual ~CubeMapTexture();
	
	/**
	* @brief Initialize cubemap data
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

protected:
	/**
	 * @brief Initialize the cubemap texture data
	 * 
	 * @param texture 
	 */
	virtual void setCubeMapTextureData(TextureData* texture);


}; 

/******************************************************************************************************************************************************************************************************************/
/**
 * @brief 
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








#endif 
