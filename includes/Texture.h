#ifndef TEXTURE_H
#define TEXTURE_H

#include "constants.h" 
#include "utils_3D.h" 
#include <QOpenGLTexture> 

class TextureData{
public:
	unsigned int width ; 
	unsigned int height ; 
	std::string name ; 
	uint32_t *data ; 
	
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
};


class Texture{
public:
	enum TYPE : unsigned {DIFFUSE = 0 , NORMAL = 1 , METALLIC = 2 , ROUGHNESS = 3 , AMBIANTOCCLUSION = 4 , SPECULARTINT= 5,  GENERIC = 6} ; 
	
	Texture(); 
	Texture(TextureData *tex); 
	virtual ~Texture();
	void set(TextureData *texture); 
	void clean();
	virtual void bindTexture() = 0 ; 
	virtual void unbindTexture() = 0;
	virtual void setGlData() = 0 ; 
	void cleanGlData(); 

protected:
	void initializeTexture2D(); 



protected:
	std::string name ;
	unsigned int width ; 
	unsigned int height ; 
	uint32_t *data ; 
	unsigned int sampler2D ; 


	
};

//Albedo texture
class DiffuseTexture : public Texture{
public:
	DiffuseTexture();
	DiffuseTexture(TextureData *data):Texture(data){}; 
	virtual ~DiffuseTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture() ; 
	virtual void unbindTexture() ;
	static const char* getTextureTypeCStr()   ; 	
}; 


class NormalTexture : public Texture{
public:
	NormalTexture();
	NormalTexture(TextureData* data):Texture(data){} ; 
	virtual ~NormalTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture() ; 
	virtual void unbindTexture() ;
	static const char* getTextureTypeCStr()  ; 	
};

class MetallicTexture : public Texture{
public:
	MetallicTexture();
	MetallicTexture(TextureData* data):Texture(data){}; 
	virtual ~MetallicTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture() ; 
	virtual void unbindTexture();
	static const char* getTextureTypeCStr()  ; 	
};

class RoughnessTexture : public Texture{
public:
	RoughnessTexture();
	RoughnessTexture(TextureData *data):Texture(data){}; 
	virtual ~RoughnessTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture()  ; 
	virtual void unbindTexture() ;
	static const char* getTextureTypeCStr() ; 	
};

class AmbiantOcclusionTexture : public Texture{
public:
	AmbiantOcclusionTexture();
	AmbiantOcclusionTexture(TextureData *data):Texture(data){}; 
	virtual ~AmbiantOcclusionTexture(); 
	virtual void setGlData() ;
	virtual void bindTexture()  ; 
	virtual void unbindTexture();
	static  const char* getTextureTypeCStr() ; 	
};
class SpecularTintTexture : public Texture{
public:
	SpecularTintTexture(); 
	SpecularTintTexture(TextureData *data):Texture(data){};
	virtual ~SpecularTintTexture(); 
	virtual void setGlData(); 
	virtual void bindTexture(); 
	virtual void unbindTexture(); 
	static const char* getTextureTypeCStr(); 
}; 

class GenericTexture : public Texture{
public:
	GenericTexture();
	GenericTexture(TextureData* data):Texture(data){}; 
	virtual ~GenericTexture(); 
	virtual void setGlData(); 
	virtual void bindTexture()  ; 
	virtual void unbindTexture() ;
	static  const char* getTextureTypeCStr() ; 	
}; 





#endif 
