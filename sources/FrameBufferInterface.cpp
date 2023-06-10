#include "../includes/FrameBufferInterface.h"

int FrameBufferInterface::setUpEmptyTexture(TextureDatabase* database, unsigned width , unsigned height){    
    TextureData temp_empty_data_texture ; 
    temp_empty_data_texture.width = width ; 
    temp_empty_data_texture.height = height ;
    temp_empty_data_texture.data = nullptr;   
    return database->addTexture(&temp_empty_data_texture , Texture::FRAMEBUFFER , true); 
}

FrameBufferInterface::FrameBufferInterface(){
    gl_framebuffer_object = nullptr; 
    texture_dim = nullptr; 
    texture_database = nullptr; 
    fbo_texture_pointer = nullptr; 
    default_framebuffer_pointer = 0 ;   
    texture_id = 0 ; 
    fbo_texture_pointer = nullptr; 
}

FrameBufferInterface::FrameBufferInterface(TextureDatabase* texture_database , ScreenSize* _texture_dim ):FrameBufferInterface(){
    texture_dim = _texture_dim ;  
    assert(texture_dim != nullptr); 
}


FrameBufferInterface::~FrameBufferInterface(){
    if(gl_framebuffer_object != nullptr)
        delete gl_framebuffer_object ;  
}

void FrameBufferInterface::resize(){
    if(fbo_texture_pointer && texture_dim && gl_framebuffer_object){
        fbo_texture_pointer->setNewSize(texture_dim->width , texture_dim->height); 
        gl_framebuffer_object->resize(texture_dim->width , texture_dim->height); 
    }
}

void FrameBufferInterface::bindFrameBuffer(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->bind(); 
}

void FrameBufferInterface::unbindFrameBuffer(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->unbind(); 
}

void FrameBufferInterface::clean(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->clean(); 
}

void FrameBufferInterface::initializeFrameBufferTexture(){
    texture_id = setUpEmptyTexture(texture_database , texture_dim->width , texture_dim->height);
    fbo_texture_pointer = texture_database->get(texture_id); 

}


void FrameBufferInterface::initializeFrameBuffer(){   
    gl_framebuffer_object = new GLFrameBuffer(texture_dim->width , texture_dim->height, fbo_texture_pointer->getSamplerID() , GLFrameBuffer::COLOR0 , GLRenderBuffer::DEPTH24_STENCIL8 , default_framebuffer_pointer); 
    gl_framebuffer_object->initializeBuffers();

}