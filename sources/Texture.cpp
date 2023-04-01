#include "../includes/Texture.h"
#include <map>



static std::map<Texture::TYPE , const char*> texture_type_c_str = {
	{Texture::DIFFUSE , "diffuse"}, 
	{Texture::NORMAL , "normal"}, 
	{Texture::METALLIC , "metallic"}, 
	{Texture::ROUGHNESS , "roughness"}, 
	{Texture::AMBIANTOCCLUSION , "ambiantocclusion"}, 
	{Texture::SPECULARTINT , "speculartint"},
	{Texture::GENERIC , "generic"}

};

Texture::Texture(){
	name = "" ; 
	width = 0 ; 
	height = 0 ; 
	data = nullptr ; 
	sampler2D = 0 ; 
}

Texture::Texture(TextureData* tex){
	name = "" ; 
	width = 0 ; 
	height = 0 ; 
	data = nullptr ; 
	sampler2D = 0 ; 
	set(tex) ; 
}


Texture::~Texture(){
	

}

void Texture::set(TextureData *texture){
	clean(); 
	name = texture->name; 
	width = texture->width ; 
	height = texture->height ; 
	data = new uint32_t [ width * height ] ; 
	for(unsigned int i = 0 ; i < width * height ; i++)
		data[i] = texture->data[i] ; 

}

void Texture::clean(){
	cleanGlData(); 
	if(data != nullptr)
		delete data; 
	data = nullptr ; 
	width = 0 ; 
	height = 0 ; 
	name = "" ; 

}

void Texture::initializeTexture2D(){
	glTexImage2D(GL_TEXTURE_2D , 0 , GL_RGBA , width , height , 0 , GL_BGRA , GL_UNSIGNED_BYTE , data); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}


void Texture::cleanGlData(){
	if(sampler2D != 0)
		glDeleteTextures(1 , &sampler2D); 
}

	

/****************************************************************************************************************************/
DiffuseTexture::DiffuseTexture(){

}

DiffuseTexture::~DiffuseTexture(){

}

void DiffuseTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + DIFFUSE); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}


void DiffuseTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + DIFFUSE); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void DiffuseTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + DIFFUSE); 
	glBindTexture(GL_TEXTURE_2D , 0);

}

const char* DiffuseTexture::getTextureTypeCStr() {
	return texture_type_c_str[DIFFUSE] ; 		
}

/****************************************************************************************************************************/
NormalTexture::NormalTexture(){

}

NormalTexture::~NormalTexture(){

}

void NormalTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + NORMAL); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 

}


void NormalTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + NORMAL); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void NormalTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + NORMAL); 
	glBindTexture(GL_TEXTURE_2D , 0); 

}

const char* NormalTexture::getTextureTypeCStr()  {
	return texture_type_c_str[NORMAL] ; 		
}




/****************************************************************************************************************************/
MetallicTexture::MetallicTexture(){


}

MetallicTexture::~MetallicTexture(){

}

void MetallicTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + METALLIC); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}


void MetallicTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + METALLIC); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void MetallicTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + METALLIC); 
	glBindTexture(GL_TEXTURE_2D , 0); 

}

const char* MetallicTexture::getTextureTypeCStr() {
	return texture_type_c_str[METALLIC] ; 		
}



/****************************************************************************************************************************/
RoughnessTexture::RoughnessTexture(){

}

RoughnessTexture::~RoughnessTexture(){

}

void RoughnessTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + ROUGHNESS); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}

void RoughnessTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + ROUGHNESS); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void RoughnessTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + ROUGHNESS); 
	glBindTexture(GL_TEXTURE_2D , 0); 

}

const char* RoughnessTexture::getTextureTypeCStr()  {
	return texture_type_c_str[ROUGHNESS] ; 		
}



/****************************************************************************************************************************/
AmbiantOcclusionTexture::AmbiantOcclusionTexture(){

}

AmbiantOcclusionTexture::~AmbiantOcclusionTexture(){

}

void AmbiantOcclusionTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}

void AmbiantOcclusionTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void AmbiantOcclusionTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION); 
	glBindTexture(GL_TEXTURE_2D , 0); 

}

const char* AmbiantOcclusionTexture::getTextureTypeCStr()  {
	return texture_type_c_str[AMBIANTOCCLUSION] ; 		
}


/****************************************************************************************************************************/
SpecularTintTexture::SpecularTintTexture(){

}

SpecularTintTexture::~SpecularTintTexture(){

}

void SpecularTintTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + SPECULARTINT); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}

void SpecularTintTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + SPECULARTINT); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void SpecularTintTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + SPECULARTINT); 
	glBindTexture(GL_TEXTURE_2D , 0); 
}


const char* SpecularTintTexture::getTextureTypeCStr() {
	return texture_type_c_str[SPECULARTINT] ; 		
}


/****************************************************************************************************************************/
GenericTexture::GenericTexture(){

}

GenericTexture::~GenericTexture(){

}

void GenericTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + GENERIC); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}

void GenericTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + GENERIC); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void GenericTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + GENERIC); 
	glBindTexture(GL_TEXTURE_2D , 0); 
}


const char* GenericTexture::getTextureTypeCStr() {
	return texture_type_c_str[GENERIC] ; 		
}


/****************************************************************************************************************************/


