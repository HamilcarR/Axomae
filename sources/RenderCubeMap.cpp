#include "../includes/RenderCubeMap.h"

RenderCubeMap::RenderCubeMap(){

}

RenderCubeMap::RenderCubeMap(TextureDatabase* database , ScreenSize* screen , unsigned int* default_fbo): FrameBufferInterface(database , screen , default_fbo ,GLFrameBuffer::COLOR0 , Texture::CUBEMAP , Texture::RGB32F , Texture::RGB , Texture::FLOAT){
    std::cout << "initialized cubemap" << std::endl;  
}


RenderCubeMap::~RenderCubeMap(){

}

void RenderCubeMap::renderToFace(unsigned i , GLFrameBuffer::INTERNAL_FORMAT color_attachment){
    gl_framebuffer_object->setColorAttachment(color_attachment);
    gl_framebuffer_object->attachTexture2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i); 
}


int RenderCubeMap::setUpEmptyTexture( unsigned width , unsigned height , Texture::TYPE type){    
    TextureData temp_empty_data_texture ; 
    temp_empty_data_texture.width = width ; 
    temp_empty_data_texture.height = height ;
    temp_empty_data_texture.data = nullptr;
    temp_empty_data_texture.f_data = nullptr;  
    temp_empty_data_texture.internal_format = internal_format ; 
    temp_empty_data_texture.data_format = data_format ; 
    temp_empty_data_texture.data_type = data_type ;   
    return texture_database->addTexture(&temp_empty_data_texture , type , false); 
}