#ifndef TEXTURE_H
#define TEXTURE_H

#include "constants.h" 
#include "utils_3D.h" 

class TextureData{
public:

	enum CHANNELS : unsigned { RGB = 0 , RGBA = 1} ; 		
	TextureData(){
		width = 0 ; 
		height = 0 ; 
		data = nullptr ; 
	}
	~TextureData(){}
	/*provides deep copy of the object , but doesn't do the cleanup for the copied object*/
	TextureData& operator=(const TextureData& from){ 
		width = from.width ;
		height = from.height ; 
		data = new uint32_t [from.width * from.height] ; 
		memcpy((void*) data , (void*) from.data , from.width * from.height * sizeof(uint32_t));		
		name = from.name ; 
		return *this ; 
	}
	void clean(){
		if(data != nullptr)
			delete data ; 
		data = nullptr ;
		width = 0 ; 
		height = 0 ;
		name = "" ; 
	}

public:
	unsigned int width ; 
	unsigned int height ; 
	std::string name ; 
	uint32_t *data ; 

};


class Texture{
public:
	enum TYPE : signed {EMPTY = -1 , DIFFUSE = 0 , NORMAL = 1 , METALLIC = 2 , ROUGHNESS = 3 , AMBIANTOCCLUSION = 4 , SPECULAR = 5, EMISSIVE = 6 , CUBEMAP = 7 , GENERIC = 8} ; 	

	Texture(); 
	Texture(TextureData *tex); 
	virtual ~Texture();
	void set(TextureData *texture); 
	void clean();
	void setTextureType(TYPE type){name = type;} 	
	TYPE getTextureType(){return name;} ;  
	virtual void bindTexture() = 0 ; 
	virtual void unbindTexture() = 0;
	virtual void setGlData() = 0 ; 
	void cleanGlData(); 

protected:
	virtual void initializeTexture2D(); 



protected:
	TYPE name ;
	unsigned int width ; 
	unsigned int height ; 
	uint32_t *data ; 
	unsigned int sampler2D ; 


	
};

//Albedo texture
class DiffuseTexture : public Texture{
public:
	DiffuseTexture();
	DiffuseTexture(TextureData *data); 
	virtual ~DiffuseTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture() ; 
	virtual void unbindTexture() ;
	static const char* getTextureTypeCStr()   ; 	
}; 


class NormalTexture : public Texture{
public:
	NormalTexture();
	NormalTexture(TextureData* data) ; 
	virtual ~NormalTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture() ; 
	virtual void unbindTexture() ;
	static const char* getTextureTypeCStr()  ; 	
};

class MetallicTexture : public Texture{
public:
	MetallicTexture();
	MetallicTexture(TextureData* data); 
	virtual ~MetallicTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture() ; 
	virtual void unbindTexture();
	static const char* getTextureTypeCStr()  ; 	
};

class RoughnessTexture : public Texture{
public:
	RoughnessTexture();
	RoughnessTexture(TextureData *data); 
	virtual ~RoughnessTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture()  ; 
	virtual void unbindTexture() ;
	static const char* getTextureTypeCStr() ; 	
};

class AmbiantOcclusionTexture : public Texture{
public:
	AmbiantOcclusionTexture();
	AmbiantOcclusionTexture(TextureData *data); 
	virtual ~AmbiantOcclusionTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture()  ; 
	virtual void unbindTexture();
	static  const char* getTextureTypeCStr() ; 	
};
class SpecularTexture : public Texture{
public:
	SpecularTexture(); 
	SpecularTexture(TextureData *data);
	virtual ~SpecularTexture(); 
	virtual void setGlData(); 
	virtual void bindTexture(); 
	virtual void unbindTexture(); 
	static const char* getTextureTypeCStr(); 
}; 

class EmissiveTexture : public Texture{
public:
	EmissiveTexture(); 
	EmissiveTexture(TextureData *data);
	virtual ~EmissiveTexture(); 
	virtual void setGlData(); 
	virtual void bindTexture(); 
	virtual void unbindTexture(); 
	static const char* getTextureTypeCStr(); 
}; 


class GenericTexture : public Texture{
public:
	GenericTexture();
	GenericTexture(TextureData* data);  
	virtual ~GenericTexture(); 
	virtual void setGlData(); 
	virtual void bindTexture()  ; 
	virtual void unbindTexture() ;
	static  const char* getTextureTypeCStr() ; 	
}; 


/*
 * We will use width * height as being the size of one single face. The total size of the cubemap will be hence : 
 * 6 * width * height * sizeof(uint32_t) bytes , with height = width .
 * 
 *     width² = RIGHT /  GL_TEXTURE_CUBE_MAP_POSITIVE_X
 * 2 * width² = LEFT / GL_TEXTURE_CUBE_MAP_NEGATIVE_X
 * 3 * width² = TOP / GL_TEXTURE_CUBE_MAP_POSITIVE_Y
 * 4 * width² = BOTTOM / GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
 * 5 * width² = BACK / GL_TEXTURE_CUBE_MAP_POSITIVE_Z
 * 6 * width² = FRONT / GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
 *
 * Data array is still 1D 
 */
class CubeMapTexture : public Texture{
public:
	CubeMapTexture();
	CubeMapTexture(TextureData* data);  
	virtual ~CubeMapTexture();
	virtual void initializeCubeMapTexture(); 
	virtual void setGlData(); 
	virtual void bindTexture()  ; 
	virtual void unbindTexture() ;
	static  const char* getTextureTypeCStr() ; 	
protected:
	virtual void setCubeMapTextureData(TextureData* texture); 

}; 

#endif 
