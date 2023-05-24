#include "../includes/Texture.h"
#include <map>

static std::map<Texture::TYPE , const char*> texture_type_c_str = {
	{Texture::DIFFUSE , "diffuse_map"}, 
	{Texture::NORMAL , "normal_map"}, 
	{Texture::METALLIC , "metallic_map"}, 
	{Texture::ROUGHNESS , "roughness_map"}, 
	{Texture::AMBIANTOCCLUSION , "ambiantocclusion_map"}, 
	{Texture::SPECULAR , "specular_map"},
	{Texture::EMISSIVE , "emissive_map"},
	{Texture::CUBEMAP , "cubemap"}, 
	{Texture::GENERIC , "generic_map"}, 
	{Texture::FRAMEBUFFER , "framebuffer_map"}

};

Texture::Texture(){
	name = EMPTY ; 
	width = 0 ; 
	height = 0 ; 
	data = nullptr ; 
	sampler2D = 0 ; 
}

Texture::Texture(TextureData* tex){
	name = EMPTY ; 
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
	name = EMPTY ; 

}

void Texture::setTextureParametersOptions(){
	glGenerateMipmap(GL_TEXTURE_2D); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER , GL_LINEAR_MIPMAP_LINEAR); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	errorCheck(); 	

}

void Texture::initializeTexture2D(){
	glTexImage2D(GL_TEXTURE_2D , 0 , GL_RGBA , width , height , 0 , GL_BGRA , GL_UNSIGNED_BYTE , data); 
	setTextureParametersOptions(); 
}


void Texture::cleanGlData(){
	if(sampler2D != 0)
		glDeleteTextures(1 , &sampler2D); 
}

	

/****************************************************************************************************************************/
DiffuseTexture::DiffuseTexture(){
	name = DIFFUSE ; 
}

DiffuseTexture::~DiffuseTexture(){

}

DiffuseTexture::DiffuseTexture(TextureData* data):Texture(data){
	name = DIFFUSE ; 
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
	name = NORMAL ; 
}

NormalTexture::~NormalTexture(){

}

NormalTexture::NormalTexture(TextureData* data):Texture(data){
	name = NORMAL ; 
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
	name = METALLIC ; 
}

MetallicTexture::~MetallicTexture(){

}

MetallicTexture::MetallicTexture(TextureData* data):Texture(data){
	name = METALLIC ; 
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
	name = ROUGHNESS ; 
}

RoughnessTexture::~RoughnessTexture(){

}

RoughnessTexture::RoughnessTexture(TextureData* data):Texture(data){
	name = ROUGHNESS ; 
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
	name = AMBIANTOCCLUSION ; 
}

AmbiantOcclusionTexture::~AmbiantOcclusionTexture(){

}

AmbiantOcclusionTexture::AmbiantOcclusionTexture(TextureData* data):Texture(data){
	name = AMBIANTOCCLUSION ; 
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
SpecularTexture::SpecularTexture(){
	name = SPECULAR ; 
}

SpecularTexture::~SpecularTexture(){

}

SpecularTexture::SpecularTexture(TextureData* data):Texture(data){
	name = SPECULAR ; 
}

void SpecularTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + SPECULAR); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}

void SpecularTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + SPECULAR); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void SpecularTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + SPECULAR); 
	glBindTexture(GL_TEXTURE_2D , 0); 
}


const char* SpecularTexture::getTextureTypeCStr() {
	return texture_type_c_str[SPECULAR] ; 		
}

/****************************************************************************************************************************/
EmissiveTexture::EmissiveTexture(){
	name = EMISSIVE ; 
}

EmissiveTexture::~EmissiveTexture(){

}

EmissiveTexture::EmissiveTexture(TextureData* data):Texture(data){
	name = EMISSIVE ; 
}

void EmissiveTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + EMISSIVE); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	Texture::initializeTexture2D(); 
}

void EmissiveTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + EMISSIVE); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void EmissiveTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + EMISSIVE); 
	glBindTexture(GL_TEXTURE_2D , 0); 
}


const char* EmissiveTexture::getTextureTypeCStr() {
	return texture_type_c_str[EMISSIVE] ; 		
}



/****************************************************************************************************************************/
GenericTexture::GenericTexture(){
	name = GENERIC ; 
}

GenericTexture::~GenericTexture(){

}

GenericTexture::GenericTexture(TextureData* data):Texture(data){
	name = GENERIC ; 
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
CubeMapTexture::CubeMapTexture(){
	name = CUBEMAP ; 
}

CubeMapTexture::~CubeMapTexture(){

}

void CubeMapTexture::setCubeMapTextureData(TextureData *texture){
	clean(); 
	width = texture->width ; 
	height = texture->height ; 
	data = new uint32_t [ width * height * 6 ] ; 
	for(unsigned int i = 0 ; i < width * height * 6 ; i++)
		data[i] = texture->data[i] ; 
}

CubeMapTexture::CubeMapTexture(TextureData* data){
	name = CUBEMAP ; 
	setCubeMapTextureData(data) ; 
}

void CubeMapTexture::initializeTexture2D(){
	for( unsigned int i = 1 ; i <= 6 ; i++){
		uint32_t* pointer_to_data = data + (i - 1) * width * height ;
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (i - 1)  , 0 , GL_RGBA , width , height , 0 , GL_RGBA , GL_UNSIGNED_BYTE , pointer_to_data); 
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);  
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER , GL_LINEAR); 
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
		errorCheck(); 	
	}

}

void CubeMapTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + CUBEMAP); 
	glBindTexture(GL_TEXTURE_CUBE_MAP , sampler2D); 
	CubeMapTexture::initializeTexture2D(); 
}

void CubeMapTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + CUBEMAP ); 
	glBindTexture(GL_TEXTURE_CUBE_MAP , sampler2D);
}

void CubeMapTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + CUBEMAP); 
	glBindTexture(GL_TEXTURE_CUBE_MAP , 0); 
}


const char* CubeMapTexture::getTextureTypeCStr() {
	return texture_type_c_str[CUBEMAP] ; 		
}

/****************************************************************************************************************************/

FrameBufferTexture::FrameBufferTexture(){
	name = FRAMEBUFFER ; 
}

FrameBufferTexture::~FrameBufferTexture(){

}

void FrameBufferTexture::setGlData(){
	glGenTextures(1 , &sampler2D); 
	glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER);
	glTexImage2D(GL_TEXTURE_2D , 0 , GL_RGBA , width , height , 0 , GL_BGRA , GL_UNSIGNED_BYTE , data);// TODO complete 
	Texture::setTextureParametersOptions(); 
}

void FrameBufferTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
}

void FrameBufferTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER); 
	glBindTexture(GL_TEXTURE_2D , 0); 
}

const char* FrameBufferTexture::getTextureTypeCStr(){
	return texture_type_c_str[FRAMEBUFFER]; 
}








