#include "../includes/Texture.h"
#include "../includes/Shader.h"
#include <map>

constexpr unsigned int DUMMY_TEXTURE_DIM = 1 ;
constexpr uint32_t DEFAULT_NORMAL_DUMMY_PIXEL_RGBA = 0x007F7FFF; //Default pixel color for a normal map 
constexpr uint32_t DEFAULT_OPACITY_DUMMY_PIXEL_RGBA = 0xFF000000;//Default pixel color for other textures 
static std::map<Texture::TYPE , const char*> texture_type_c_str = {
	{Texture::DIFFUSE , "diffuse_map"}, 
	{Texture::NORMAL , "normal_map"}, 
	{Texture::METALLIC , "metallic_map"}, 
	{Texture::ROUGHNESS , "roughness_map"}, 
	{Texture::AMBIANTOCCLUSION , "ambiantocclusion_map"}, 
	{Texture::SPECULAR , "specular_map"},
	{Texture::EMISSIVE , "emissive_map"},
	{Texture::OPACITY , "opacity_map"}, 
	{Texture::CUBEMAP , "cubemap"},
	{Texture::ENVMAP  , "environment_map"}, 
	{Texture::GENERIC , "generic_map"}, 
	{Texture::FRAMEBUFFER , "framebuffer_map"}

};

/**
 * The function sets the data of a dummy texture to a solid black color.
 * 
 * @param dummy The parameter "dummy" is a pointer to a struct of type "TextureData". The function sets
 * the width, height, and data of the TextureData struct pointed to by "dummy" to create a dummy
 * texture with a solid black color.
 */
static void set_dummy_TextureData(TextureData* dummy){
	dummy->width = DUMMY_TEXTURE_DIM; 
	dummy->height = DUMMY_TEXTURE_DIM ;
	dummy->data = new uint32_t[dummy->width * dummy->height];  
	for(unsigned i = 0 ; i < dummy->width * dummy->height ; i++){
		dummy->data[i] = DEFAULT_OPACITY_DUMMY_PIXEL_RGBA ; 
	}	
}

/**
 * The function sets the width, height, and data of a dummy texture with a constant blue color , 
 * indicating a uniform vector (0 , 0 , 1) representing the normal. 
 * 
 * @param dummy A pointer to a TextureData struct that will be filled with one only pixel having the value (128 , 128 , 255).
 */

static void set_dummy_TextureData_normals(TextureData* dummy){
	dummy->width = DUMMY_TEXTURE_DIM; 
	dummy->height = DUMMY_TEXTURE_DIM ; 
	dummy->data = new uint32_t[dummy->width * dummy->height];  
	for(unsigned i = 0 ; i < DUMMY_TEXTURE_DIM * DUMMY_TEXTURE_DIM; i++)
		dummy->data[i] = DEFAULT_NORMAL_DUMMY_PIXEL_RGBA ; 
}


Texture::Texture(){
	name = EMPTY ;
	is_dummy = false ;  
	width = 0 ; 
	height = 0 ; 
	data = nullptr ;
	f_data = nullptr;  
	sampler2D = 0 ; 
	internal_format = RGBA; 
	data_format = BGRA ; 
}

Texture::Texture(TextureData* tex):Texture(){
	if(tex != nullptr)	
		set(tex) ;
}

Texture::~Texture(){

}

void Texture::set(TextureData *texture){
	clean(); 
	width = texture->width ; 
	height = texture->height ;
	if(texture->data){
		data = new uint32_t [ width * height ] ; 
		for(unsigned int i = 0 ; i < width * height ; i++)
			data[i] = texture->data[i] ; 
	}
	if(texture->f_data){
		f_data = new float [width * height * texture->nb_components] ; 
		for(unsigned i = 0 ; i < width * height * texture->nb_components ; i++)
			f_data[i] = texture->f_data[i] ; 
	}
	data_format = static_cast<Texture::FORMAT>(texture->data_format) ; 
	internal_format = static_cast<Texture::FORMAT>(texture->internal_format) ; 
	data_type = static_cast<Texture::FORMAT>(texture->data_type) ; 
}

void Texture::clean(){
	cleanGlData(); 
	if(data != nullptr)
		delete data;
	if(f_data)
		delete f_data ; 
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
	errorCheck(__FILE__ , __LINE__); 	
}

void Texture::initializeTexture2D(){
	glTexImage2D(GL_TEXTURE_2D , 0 , internal_format, width , height , 0 , data_format , data_type , data); 
	setTextureParametersOptions(); 
}

void Texture::setNewSize(unsigned _width , unsigned _height){
	width = _width ; 
	height = _height; 
	bindTexture(); 
	glTexImage2D(GL_TEXTURE_2D , 0 , internal_format , _width , _height , 0 , data_format , data_type , nullptr); 
	unbindTexture(); 
}

void Texture::cleanGlData(){
	if(sampler2D != 0){
		glDeleteTextures(1 , &sampler2D);
		sampler2D = 0 ;  
	}
}

/****************************************************************************************************************************/
DiffuseTexture::DiffuseTexture(){
	name = DIFFUSE ; 
}

DiffuseTexture::~DiffuseTexture(){

}

DiffuseTexture::DiffuseTexture(TextureData* data){
	name = DIFFUSE ; 
	if(data != nullptr)
		set(data); 
}

void DiffuseTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + DIFFUSE); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D();
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
}


void DiffuseTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + DIFFUSE); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void DiffuseTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + DIFFUSE); 
	glBindTexture(GL_TEXTURE_2D , 0);

}


void DiffuseTexture::set(TextureData *texture){
	clean(); 
	width = texture->width ; 
	height = texture->height ; 
	data = new uint32_t [ width * height ] ;
	data_format = static_cast<Texture::FORMAT>(texture->data_format) ; 
	internal_format = static_cast<Texture::FORMAT>(texture->internal_format) ; 
	data_type = static_cast<Texture::FORMAT>(texture->data_type) ; 
	has_transparency = false ; 
	for(unsigned int i = 0 ; i < width * height ; i++){
		data[i] = texture->data[i] ;
		if((data[i] & 0xFF000000) != 0xFF000000)
			has_transparency = true ; 
	}
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

NormalTexture::NormalTexture(TextureData* texture):Texture(texture){
	name = NORMAL ;
	
}

void NormalTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + NORMAL); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D();
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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

void MetallicTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + METALLIC); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D();
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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

void RoughnessTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + ROUGHNESS); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D();
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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

void AmbiantOcclusionTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + AMBIANTOCCLUSION); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D();
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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

void SpecularTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + SPECULAR); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D(); 
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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

void EmissiveTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + EMISSIVE); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D();
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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
OpacityTexture::OpacityTexture(){
	name = OPACITY ; 
}

OpacityTexture::~OpacityTexture(){

}

OpacityTexture::OpacityTexture(TextureData* data):Texture(data){
	name = OPACITY ; 
}

void OpacityTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + OPACITY); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D();
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
}

void OpacityTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + OPACITY); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
}

void OpacityTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + OPACITY); 
	glBindTexture(GL_TEXTURE_2D , 0); 
}


const char* OpacityTexture::getTextureTypeCStr() {
	return texture_type_c_str[OPACITY] ; 		
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


void GenericTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + GENERIC); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
	if(data != nullptr)
		Texture::initializeTexture2D(); 
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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
CubeMapTexture::CubeMapTexture(FORMAT _internal_format , FORMAT _data_format , FORMAT _data_type , unsigned _width , unsigned _height):Texture(){ //! Move arguments to Texture()
	name = CUBEMAP ; 
	internal_format = _internal_format ; 
	data_format = _data_format ; 
	data_type = _data_type;
	width = _width ; 
	height = _height ;  
}

CubeMapTexture::~CubeMapTexture(){

}

void CubeMapTexture::setCubeMapTextureData(TextureData *texture){
	clean(); 
	width = texture->width ; 
	height = texture->height ;
	internal_format = static_cast<Texture::FORMAT> (texture->internal_format); 
	data_format = static_cast<Texture::FORMAT> (texture->data_format) ; 
	data_type = static_cast<Texture::FORMAT> (texture->data_type);	
	if(texture->data){
		data = new uint32_t [ width * height * 6 ] ; 
		for(unsigned int i = 0 ; i < width * height * 6 ; i++)
			data[i] = texture->data[i] ; 
	}
}

CubeMapTexture::CubeMapTexture(TextureData* data):CubeMapTexture(){
	name = CUBEMAP ;
	if(data)
		setCubeMapTextureData(data) ; 
}

void CubeMapTexture::initializeTexture2D(){
	if(data)
		for( unsigned int i = 1 ; i <= 6 ; i++){
			uint32_t* pointer_to_data = data + (i - 1) * width * height ;
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (i - 1)  , 0 , internal_format , width , height , 0 , data_format , data_type , pointer_to_data); 
			errorCheck(__FILE__ , __LINE__); 	
		}
	else
		for(unsigned i = 0 ; i < 6 ; i++){
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i , 0 , internal_format , width , height , 0 , data_format , data_type , nullptr); 
			errorCheck(__FILE__ , __LINE__); 
		}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);  
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER , GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
}

void CubeMapTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + CUBEMAP); 
	glBindTexture(GL_TEXTURE_CUBE_MAP , sampler2D); 
	CubeMapTexture::initializeTexture2D(); 
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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
/* This is loaded from the disk as an .hdr image , with 4 bytes float for texel format*/

EnvironmentMapTexture::EnvironmentMapTexture(FORMAT _internal_format  , FORMAT _data_format  , FORMAT _data_type , unsigned _width  , unsigned _height ) : Texture(){
	name = ENVMAP;
	internal_format = _internal_format ; 
	data_format =_data_format ;
	data_type = _data_type;
	width = _width ; 
	height = _height ;   
}

EnvironmentMapTexture::EnvironmentMapTexture(TextureData* data) : Texture(data){
	name = ENVMAP ; 
	
}

EnvironmentMapTexture::~EnvironmentMapTexture(){

}

void EnvironmentMapTexture::initializeTexture2D(){
	glTexImage2D(GL_TEXTURE_2D , 0 , internal_format , width , height , 0 , data_format , data_type , f_data); 
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER , GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D); 
	errorCheck(__FILE__ , __LINE__); 	
}

void EnvironmentMapTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 	
	glActiveTexture(GL_TEXTURE0 + ENVMAP); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
	if(f_data != nullptr)
		EnvironmentMapTexture::initializeTexture2D(); 
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
}

void EnvironmentMapTexture::bindTexture(){
	glActiveTexture(GL_TEXTURE0 + ENVMAP); 
	glBindTexture(GL_TEXTURE_2D , sampler2D); 
}

void EnvironmentMapTexture::unbindTexture(){
	glActiveTexture(GL_TEXTURE0 + ENVMAP); 
	glBindTexture(GL_TEXTURE_2D , 0); 
}

const char* EnvironmentMapTexture::getTextureTypeCStr(){
	return texture_type_c_str[ENVMAP] ;
}


/****************************************************************************************************************************/

FrameBufferTexture::FrameBufferTexture():Texture(){
	name = FRAMEBUFFER ;
	internal_format = RGBA16F ; 
	data_format = BGRA ; 
	data_type = UBYTE ;   
}

FrameBufferTexture::~FrameBufferTexture(){

}

FrameBufferTexture::FrameBufferTexture(TextureData *_data):FrameBufferTexture(){
	if(_data != nullptr){
		width = _data->width; 
		height = _data->height;
		internal_format = static_cast<Texture::FORMAT>(_data->internal_format); 
		data_format = static_cast<Texture::FORMAT>(_data->data_format); 
		data_type = static_cast<Texture::FORMAT>(_data->data_type); 
		this->data = nullptr; 
	}
}

FrameBufferTexture::FrameBufferTexture(unsigned _width , unsigned _height):FrameBufferTexture(){
	width = _width ; 
	height = _height ; 
}

void FrameBufferTexture::initializeTexture2D(){
	glTexImage2D(GL_TEXTURE_2D , 0 , internal_format , width , height , 0 , data_format , data_type , nullptr);
	glTexParameteri(GL_TEXTURE_2D , GL_TEXTURE_MIN_FILTER , GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_2D , GL_TEXTURE_MAG_FILTER , GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D , GL_TEXTURE_WRAP_S , GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D , GL_TEXTURE_WRAP_T , GL_CLAMP_TO_EDGE); 
}

void FrameBufferTexture::setGlData(Shader* shader){
	glGenTextures(1 , &sampler2D); 
	glActiveTexture(GL_TEXTURE0 + FRAMEBUFFER); 
	glBindTexture(GL_TEXTURE_2D , sampler2D);
	FrameBufferTexture::initializeTexture2D(); 
	shader->setTextureUniforms(texture_type_c_str[name] , name);  
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

/****************************************************************************************************************************/
DummyDiffuseTexture::DummyDiffuseTexture(){
	name = DIFFUSE ;
	TextureData dummy ; 
	set_dummy_TextureData(&dummy); 	
	set(&dummy) ;
	dummy.clean();
	is_dummy = true ; 
}

DummyDiffuseTexture::~DummyDiffuseTexture(){

}
/****************************************************************************************************************************/
DummyNormalTexture::DummyNormalTexture(){
	name = NORMAL;
	TextureData dummy ; 
	set_dummy_TextureData_normals(&dummy); 
	set(&dummy);  
	dummy.clean() ;
	is_dummy = true ; 
}

DummyNormalTexture::~DummyNormalTexture(){

}

/****************************************************************************************************************************/
DummyMetallicTexture::DummyMetallicTexture(){
	name = METALLIC ;
	TextureData dummy ; 
	set_dummy_TextureData(&dummy); 	
	set(&dummy) ; 
	dummy.clean() ;
	is_dummy = true ; 
}

DummyMetallicTexture::~DummyMetallicTexture(){

}

/****************************************************************************************************************************/
DummyRoughnessTexture::DummyRoughnessTexture(){
	name = ROUGHNESS ; 
	TextureData dummy ; 
	set_dummy_TextureData(&dummy); 	
	set(&dummy) ; 
	dummy.clean() ;
	is_dummy = true ; 
}

DummyRoughnessTexture::~DummyRoughnessTexture(){

}

/****************************************************************************************************************************/
DummySpecularTexture::DummySpecularTexture(){
	name = SPECULAR ; 
	TextureData dummy ; 
	set_dummy_TextureData(&dummy); 	
	set(&dummy) ; 
	dummy.clean(); 
	is_dummy = true ; 
}

DummySpecularTexture::~DummySpecularTexture(){

}

/****************************************************************************************************************************/
DummyAmbiantOcclusionTexture::DummyAmbiantOcclusionTexture(){
	name = AMBIANTOCCLUSION ; 
	TextureData dummy ; 
	set_dummy_TextureData(&dummy); 	
	set(&dummy) ; 
	dummy.clean() ;
	is_dummy = true ; 
}

DummyAmbiantOcclusionTexture::~DummyAmbiantOcclusionTexture(){

}

/****************************************************************************************************************************/
DummyEmissiveTexture::DummyEmissiveTexture(){
	name = EMISSIVE ; 
	TextureData dummy ; 
	set_dummy_TextureData(&dummy); 	
	set(&dummy) ; 
	dummy.clean() ;
	is_dummy = true ; 
}

DummyEmissiveTexture::~DummyEmissiveTexture(){

}

/****************************************************************************************************************************/
DummyOpacityTexture::DummyOpacityTexture(){
	name = OPACITY ; 
	TextureData dummy ; 
	set_dummy_TextureData(&dummy); 	
	set(&dummy) ; 
	dummy.clean() ;
	is_dummy = true ; 
}

DummyOpacityTexture::~DummyOpacityTexture(){

}








































