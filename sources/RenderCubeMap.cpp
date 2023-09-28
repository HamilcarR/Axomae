#include "../includes/RenderCubeMap.h"

RenderCubeMap::RenderCubeMap(){

}

RenderCubeMap::RenderCubeMap(TextureDatabase* database , ScreenSize* screen , unsigned int* default_fbo): FrameBufferInterface(database , screen , default_fbo  ){
    LOG("initialized cubemap" , LogLevel::INFO);  
}


RenderCubeMap::~RenderCubeMap(){

}

void RenderCubeMap::renderToTexture(unsigned i , GLFrameBuffer::INTERNAL_FORMAT color_attachment , unsigned mip_level){
    Texture* tex = fbo_attachment_texture_collection[color_attachment]; 
    if(tex && tex->isInitialized())
        gl_framebuffer_object->attachTexture2D(color_attachment , static_cast<GLFrameBuffer::TEXTURE_TARGET>(GLFrameBuffer::CUBEMAP_POSITIVE_X + i) , tex->getSamplerID() , mip_level); 
}

